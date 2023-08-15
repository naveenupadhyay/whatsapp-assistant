import os
import openai
import requests as network
import constants as const
import cohere
import uuid
from fastapi import FastAPI, Request, BackgroundTasks
from twilio.rest import Client
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import SystemMessage, HumanMessage
from qdrant_client import QdrantClient, models
from langchain.document_transformers import DoctranQATransformer
from langchain.chains import LLMChain
from typing import List, Dict
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
sid = os.getenv("SID")
token = os.getenv("TOKEN")
apiKey = os.getenv("OPENAI_API_KEY")
cohereApiKey = os.getenv("COHERE_API_KEY")

# load the essential models

client = Client(sid, token)
chat_model = ChatOpenAI(openai_api_key=apiKey, model_name="gpt-4")
intent_model = cohere.Client(cohereApiKey)
embedding_model = cohere.Client(cohereApiKey)
qa_model = OpenAI(openai_api_key=apiKey, model_name="gpt-4")
qa_transformer = DoctranQATransformer(
    openai_api_key=apiKey, openai_api_model="gpt-3.5-turbo-16k"
)
# initialize the vector db
db = QdrantClient(path="app/data/kb")

# load the chains
qa_chain: LLMChain


def write_log(message: str):
    if os.path.exists("app/logs/log.txt"):
        with open("app/logs/log.txt", "a") as f:
            f.write(message + "\n")
    else:
        with open("app/logs/log.txt", "w") as f:
            f.write(message + "\n")


def embed_query(query: str):
    return embedding_model.embed(texts=[query]).embeddings[0]


def get_user_intent(user_query: str):
    response = intent_model.classify(
        inputs=[user_query], examples=const.examples, model="large"
    )
    return response.classifications[0].prediction


def create_collection():
    db.recreate_collection(
        collection_name=const.collection_name,
        vectors_config=models.VectorParams(
            size=const.COHERE_SIZE_VECTOR, distance=models.Distance.COSINE
        ),
    )


def get_generated_image(user_prompt: str):
    response = openai.Image.create(
        api_key=apiKey, prompt=user_prompt, n=1, size=const.image_size
    )
    image_url = response["data"][0]["url"]
    return image_url


def load_qa_chain(example_list: List[Dict[str, str]]):
    global qa_chain

    example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template=const.example_template
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=example_list,
        example_prompt=example_prompt,
        prefix=const.prefix,
        suffix=const.suffix,
        input_variables=["user_query"],
        example_separator="\n",
    )

    qa_chain = LLMChain(llm=qa_model, prompt=few_shot_prompt)


async def ingest_uploaded_pdf(
    msg_from: str, msg_to: str, url: str, worker: BackgroundTasks
):
    write_log(f"Starting to ingest the pdf")
    info = network.head(url=url)
    # get the expected size in mb
    expected_size = int(info.headers["Content-Length"]) / 1000000
    write_log(f"The size of the uploaded pdf {expected_size}")
    if expected_size > 2:
        client.messages.create(
            to=msg_from, from_=msg_to, body=const.size_greater_than_2
        )
    else:
        pdf_data = network.get(url)
        # write the uploaded pdf to the local storage in data folder
        with open("app/data/uploaded.pdf", "wb") as f:
            f.write(pdf_data.content)
        loader = PyPDFLoader(file_path="app/data/uploaded.pdf")
        documents = loader.load()
        transformed_docs = await qa_transformer.atransform_documents(documents)
        for doc in transformed_docs:
            qa_list = doc.metadata["questions_and_answers"]
            points = get_points(qa_list)
            db.upsert(collection_name=const.collection_name, points=points)
        worker.add_task(remove_uploaded_file, "app/data/uploaded.pdf")
        client.messages.create(to=msg_from, from_=msg_to,
                               body=const.pdf_ready_for_qa)


def float_vector(vector: List[float]):
    return list(map(float, vector))


def get_points(qa_list: List[Dict[str, str]]):
    points = [
        models.PointStruct(
            id=uuid.uuid4().hex,
            payload={"question": point["question"], "answer": point["answer"]},
            vector=float_vector(embed_query(point["question"])),
        )
        for point in qa_list
    ]
    return points


@app.on_event("startup")
async def load_doc_pipeline():
    create_collection()
    loader = PyPDFLoader("app/data/medical_insurance.pdf")
    docs = loader.load()
    transformed_document = await qa_transformer.atransform_documents(docs)
    qa_meta = transformed_document[0].metadata
    qa_list = qa_meta["questions_and_answers"]
    points = get_points(qa_list)
    # update or insert data into the database
    db.upsert(collection_name=const.collection_name, points=points)


@app.on_event("shutdown")
async def unload_pipeline():
    pass


async def get_transcript(recording_url: str):
    audio_result = network.get(recording_url)
    open("app/data/user_query.mp3", "wb").write(audio_result.content)
    audio_file = open("app/data/user_query.mp3", "rb")
    with get_openai_callback() as cb:
        transcription = openai.Audio.transcribe(
            "whisper-1", audio_file, api_key=apiKey)
        print(f"The cost for transcription is {cb.total_cost}")
    user_query = transcription["text"]
    return user_query


def remove_uploaded_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)


def run_insurance_query(msg_from: str, msg_to: str, user_query: str):
    print("Running db query")
    hits = db.search(
        collection_name=const.collection_name,
        query_vector=embed_query(user_query),
        limit=3,
    )
    examples = [hit.payload for hit in hits]
    load_qa_chain(example_list=examples)
    with get_openai_callback() as cb:
        response = qa_chain.run({"user_query": user_query})
        print(f"Cost for insurance query {cb.total_cost}")
    client.messages.create(to=msg_from, from_=msg_to, body=response)


def run_normal_query(msg_from: str, msg_to: str, user_query: str):
    print("Running normal query")
    messages = [
        SystemMessage(
            content="You are a helpful assistant, your task is to answer the query in the most appropriate way"
        ),
        HumanMessage(content=user_query),
    ]
    with get_openai_callback() as cb:
        result = chat_model(messages)
        print(f"The cost for chat is {cb.total_cost}")
    content = result.content
    chunks = [content[i: i + 1500] for i in range(0, len(content), 1500)]
    for content_body in chunks:
        client.messages.create(to=msg_from, from_=msg_to, body=content_body)


@app.get("/")
def root(worker: BackgroundTasks):
    worker.add_task(write_log, "Bot Started")
    return "Bot Started"


def run_image_query(msg_from, msg_to, user_query):
    image_url = get_generated_image(user_prompt=user_query)
    client.messages.create(to=msg_from, from_=msg_to, media_url=image_url)


def run_user_query(
    msg_from: str, msg_to: str, user_query: str, worker: BackgroundTasks
):
    worker.add_task(write_log, f"Query in Transcript: {user_query}")
    intent = get_user_intent(user_query).lower()
    worker.add_task(write_log, message=f"the intent of the user is {intent}")
    if intent in ["insurance", "pdf"]:
        run_insurance_query(msg_from, msg_to, user_query)
    elif intent == "image":
        run_image_query(msg_from, msg_to, user_query)
    else:
        run_normal_query(msg_from, msg_to, user_query)


@app.post("/hook")
async def bot(request: Request, worker: BackgroundTasks):
    data = await request.form()
    msg_to = data["To"]
    msg_from = data["From"]
    print(f"Message from {msg_from} to {msg_to}")
    # user input
    if int(data["NumMedia"]) == 1:
        print(f"Media is of type {data['MediaContentType0']}")
        # if the content type is audio then we need to transcribe it and then process the user query
        if data["MediaContentType0"] == "audio/ogg":
            audio_url = data["MediaUrl0"]
            user_query = await get_transcript(audio_url)
            worker.add_task(remove_uploaded_file, "user_query.mp3")
            run_user_query(msg_from, msg_to, user_query, worker)
        # if the content type is pdf then process the pdf, store it in a vector database and then let the user ask
        # query to the pdf
        elif data["MediaContentType0"] == "application/pdf":
            client.messages.create(
                to=msg_from,
                from_=msg_to,
                body=const.processing_pdf,
            )
            worker.add_task(
                ingest_uploaded_pdf, msg_from, msg_to, data["MediaUrl0"], worker
            )
        else:
            client.messages.create(
                to=msg_from,
                from_=msg_to,
                body=const.media_type_error,
            )
    elif int(data["NumMedia"]) > 1:
        client.messages.create(
            to=msg_from, from_=msg_to, body=const.multiple_media_error
        )
    else:
        user_query = data["Body"].lower()
        run_user_query(msg_from, msg_to, user_query, worker)

    return "Ok"
