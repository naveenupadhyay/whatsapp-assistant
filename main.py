import os
import json
import openai
import requests as network
import constants as const
from fastapi import FastAPI, Request, BackgroundTasks
from twilio.rest import Client
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.vectorstores import DeepLake
from langchain.document_transformers import DoctranQATransformer
from langchain.chains import LLMChain
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
sid = os.getenv('TWILIO_SID')
token = os.getenv('TWILIO_TOKEN')
apiKey = os.getenv('OPENAI_API_KEY')
alToken = os.getenv('ACTIVELOOP_TOKEN')
dataset_name = "medical_doc_ds"
dataset_path = f"data/{dataset_name}"

# load the essential models

client = Client(sid, token)
chat_model = ChatOpenAI(openai_api_key=apiKey, model_name="gpt-4")
intent_model = OpenAI(openai_api_key=apiKey, model_name="gpt-4")
qa_model = OpenAI(openai_api_key=apiKey, model_name="gpt-4")

# load the chains
intent_chain: LLMChain
qa_chain: LLMChain
# load the vector db
db: DeepLake
# load the embeddings
embeddings: OpenAIEmbeddings


def write_log(message: str):
    if os.path.exists("logs/log.txt"):
        with open("logs/log.txt", "a") as f:
            f.write(message + "\n")
    else:
        with open("logs/log.txt", "w") as f:
            f.write(message + "\n")


def load_intent_chain():
    global intent_chain
    intent_prompt_template = PromptTemplate(
        input_variables=["query"], template=const.DETECT_INTENT_PROMPT
    )
    intent_chain = LLMChain(llm=intent_model, prompt=intent_prompt_template)


def load_qa_chain():
    global qa_chain
    with open("data/converted_json.txt", "r") as f:
        content = json.load(f)

    example_list = content["questions_and_answers"]

    example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template=const.example_template
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        example_list, embeddings, db, k=3
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=const.prefix,
        suffix=const.suffix,
        input_variables=["user_query"],
        example_separator="\n",
    )
    qa_chain = LLMChain(llm=qa_model, prompt=few_shot_prompt)


@app.on_event("startup")
async def load_doc_pipeline():
    global db
    global embeddings
    db = DeepLake(token=alToken, dataset_path=dataset_path)
    embeddings = OpenAIEmbeddings(openai_api_key=apiKey, model="text-embedding-ada-002")
    loader = PyPDFLoader("data/medical_insurance.pdf")
    docs = loader.load()
    qa_transformer = DoctranQATransformer(
        openai_api_key=apiKey, openai_api_model="gpt-3.5-turbo-16k"
    )
    transformed_document = await qa_transformer.atransform_documents(docs)
    qa_meta = transformed_document[0].metadata
    with open("data/converted_json.txt", "w") as file:
        file.write(json.dumps(qa_meta))
    load_intent_chain()
    load_qa_chain()


@app.on_event("shutdown")
async def unload_pipeline():
    pass


async def get_transcript(recording_url: str):
    audio_result = network.get(recording_url)
    open("user_query.mp3", "wb").write(audio_result.content)
    audio_file = open("user_query.mp3", "rb")
    with get_openai_callback() as cb:
        transcription = openai.Audio.transcribe("whisper-1", audio_file, api_key=apiKey)
        print(f"The cost for transcription is {cb.total_cost}")
    user_query = transcription["text"]
    return user_query


def remove_audio_file():
    if os.path.exists("user_query.mp3"):
        os.remove("user_query.mp3")


def run_insurance_query(msg_from: str, msg_to: str, user_query: str):
    print("Running insurance query")
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
    chunks = [content[i : i + 1500] for i in range(0, len(content), 1500)]
    for content_body in chunks:
        client.messages.create(to=msg_from, from_=msg_to, body=content_body)


@app.get("/")
def root(worker: BackgroundTasks):
    worker.add_task(write_log, "Bot Started")
    return "Bot Started"


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
            worker.add_task(remove_audio_file)
            worker.add_task(write_log, f"Query in Transcript: {user_query}")
            intent = str(intent_chain.run({"query": user_query})).lower()
            if "insurance" in intent:
                run_insurance_query(msg_from, msg_to, user_query)
            else:
                run_normal_query(msg_from, msg_to, user_query)
            print(user_query)
        # if the content type is pdf then process the pdf, store it in a vector database and then let the user ask
        # query to the pdf
        elif data["MediaContentType0"] == "application/pdf":
            client.messages.create(
                to=msg_from,
                from_=msg_to,
                body="We are building PDF support, Kindly try with normal query",
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
        worker.add_task(write_log, message=f"Query in text message: {user_query}")
        with get_openai_callback() as cb:
            intent = str(intent_chain.run({"query": user_query})).lower()
            print(f"Cost for intent classification is {cb.total_cost}")
        print(f"the intent of the user is {intent}")
        worker.add_task(write_log, message=f"the intent of the user is {intent}")
        if "insurance" in intent:
            run_insurance_query(msg_from, msg_to, user_query)
        else:
            run_normal_query(msg_from, msg_to, user_query)

    return "Ok"
