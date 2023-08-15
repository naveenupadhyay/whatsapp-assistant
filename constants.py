from cohere.responses.classify import Example

prefix = """
    Reminder: You're Helpful AI assistant.
    Your goal is to help the user in their question related pdf they have recently uploaded.
    Using conversation examples, you'll help create responses and guide the user (labeled You).
    Keep your responses helpful, concise, and relevant to the conversation.
    The examples may be fragmented, incomplete, or even incorrect. Do not ask for clarification, do your best to understand what
    the examples say based on context. Be sure of everything you say.
    Keep responses concise and to the point. Here are a few helpful examples:
"""

example_template = """
    User: {question}
    AI: {answer}
"""

suffix = """
    User: {user_query}
    AI: """

DETECT_INTENT_PROMPT = """
    You are a medical insurance expert
    Your task is to read the query and discern whether the customer is a insurance related question or not.
    Do not add, infer, or interpret anything.
    Example:
    '''
    Query: What is the cost of this policy ? 
    You: insurance

    Query: Hi
    You: None

    Query: What will be the total cost in case of type 2 diabetes ?
    You: insurance
    '''
    If the query is not related to insurance, respond with 'None'.
    Starting now, you will respond only with either insurance or None: 
    {query}
"""

media_type_error = """
Currently supported media:
    1. Audio File
    2. PDF (size limit 2mb)
"""

multiple_media_error = """
We are working on supporting more than one media file at once 
Currently supported media:
    1. Audio File
    2. PDF (size limit 2mb)
"""

processing_pdf = """
We are adding the PDF to our knowledge base,
You will receive a ping once the processing is done
"""

size_greater_than_2 = """
Uploaded file exceeds the file size limit of 2mb
"""

pdf_ready_for_qa = """
Your pdf is ready for Q&A you can ask questions now related to the pdf
Note: Make sure to refer to the uploaded pdf during Q&A
Example: What does the author trying to say in the above pdf ?
"""
examples = [
    Example("What is the cost of this policy ?", "insurance"),
    Example("Hi", "general"),
    Example("Hi, How are you", "general"),
    Example(
        "can you create an image for a close up, studio photographic portrait of a white siamese cat that looks "
        "curious, backlit ears",
        "image",
    ),
    Example(
        "what is the user say in the introduction in the above uploaded pdf", "pdf"
    ),
    Example("What will be the total cost in case of type 2 diabetes ?", "insurance"),
    Example("What is the best place to visit in london", "general"),
    Example("What is the summary of the above pdf ", "pdf"),
    Example("Generate a image of a cat at the moon", "image"),
    Example("I have minor fracture how can this policy help", "insurance"),
    Example("Share an image of a man jumping from a mountain", "image"),
    Example("What is the capital of New Delhi", "general"),
    Example("I am about to have a baby what all is covered in the policy", "insurance"),
    Example("Show me an image of a black dog", "image"),
    Example("Based on the pdf just uploaded answer the following question", "pdf"),
    Example("can you send me a image of a white siamese cat", "image"),
]

# cohere vector size for large model
COHERE_SIZE_VECTOR = 4096

# collection name
collection_name = "knowledge_base"

# image size to generate
image_size = "512x512"
