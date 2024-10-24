from crewai_tools import tool
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path
from vlm_ocr import openai_read
import json

## helper functions
def read_pdf(file_path):
    text = openai_read(file_path)
    return text

def chunker(text, chunk_size=512):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

## tools
@tool('code generator')
def code_generator(goal: str, optional_context: str = None) -> str:
    """
    Generates code for a given goal and outputs a JSON string. 
    Additional context may be given to assist.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    messages = [
        (
            'system',
            '''
            You are a expert software engineer and are tasked with writing a Python script.
            This script is for a critical function within the company.
            To create the script think it through step-by-step and write the code accordingly.
            You must output a valid Python script with working syntax in a valid JSON string.

            Expected Output:
            {
                'goal': 'some goal',
                'steps': ['step 1: ...', 'step 2: ...'],
                'code': ['x=25', 'y=50', 'return x + y']
            }
            '''
        ),
        (
            'human',
            f'''
            Write a script to accomplish {goal}.
            the output must be in JSON format.

            Optional Context:
            {optional_context}
            '''
        )
    ]

    return llm.invoke(messages).content

@tool('code validator')
def code_validator(obj: str) -> str:
    """
    validates code within a given string containing a json object
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    messages = [
        (
            'system',
            '''
            You are a expert QA engineer and are tasked with validating a Python script.
            This script is for a critical function within the company.
            To validate the script think it through step-by-step and make suggestions accordingly.
            You must output a valid JSON string.

            Expected Output:
            {
                'goal': 'some goal',
                'steps': ['step 1: ...', 'step 2: ...'],
                'code': ['x=25', 'y="50"', 'return x + y'],
                'suggestions': ['variable y must be an int to add it to the int x']
            }
            '''
        ),
        (
            'human',
            f'''
            Validate the code within {obj} and write down the suggestions.
            the output must be in JSON format.
            '''
        )
    ]

    return llm.invoke(messages).content

@tool('json validator')
def json_validator(obj: str) -> str:
    """
    Validates a JSON string to ensure it is in valid JSON format
    """
    try:
        json.loads(obj)
        return obj
    except Exception as e:
        return f'The JSON string was formatted incorrectly, exception occurred: {e}'
    
@tool('embed_pdf')
def embed_pdf():
    '''
    Takes in a PDF and embeds it into Pinecone for future context usage
    '''
    pc = Pinecone()
    index_name = 'pdf-embeddings'

    exists = False
    for index in pc.list_indexes():
        if index_name == index['name']:
            exists = True

    if exists == False:
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(index_name)

    embeddings = CohereEmbeddings(
        model='embed-english-v3.0',
    )

    file_path=input('Input filepath to documentation: ')

    file_exists = Path(file_path)

    if file_exists:
        name = file_path.split('/')[-1]
        name = name.split('.')[0]
        text = read_pdf(file_path)
    
    else:
        return 'Invalid file path'

    chunks = chunker(text)
    ids = [f'{name}_chunk_{i}' for i in range(len(chunks))]

    embeddings_result = embeddings.embed_documents(
        chunks
    )
    
    vectors = [(ids[i], embeddings_result[i]) for i in range(len(embeddings_result))]
    index.upsert(vectors=vectors)


@tool('retriever')
def retriever(query: str) -> list:
    '''
    For advanced or niche questions, takes in a question and returns the relevant documents to answer
    '''
    pc = Pinecone()
    index_name = 'pdf-embeddings'

    exists = False
    for index in pc.list_indexes():
        if index_name == index['name']:
            exists = True

    if exists == False:
        return 'Index does not exist, upload a PDF for search first.'
    
    embeddings = CohereEmbeddings(
        model='embed-english-v3.0',
    )

    embedded_query = embeddings.embed_query(query)

    index = pc.Index(index_name)
    search = index.query(
        vector=embedded_query,
        top_k=3
    )

    return search['matches']

@tool('query_assessment')
def assessment(script: str) -> str:
    '''
    Decides whether or not the script is completeable without additional documentation
    '''
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    messages = [
        (
            'system',
            '''
            You are a expert software engineer and are tasked with assessing a script request's complexity.
            This script is for a critical function within the company.
            Think through the steps required to create the script, and assess if it can be done with general knowledge.
            If the script can be created with no additional documentation, output 'Simple'.
            Otherwise, output 'More documentation required'

            Expected Output:
            'More documentation required'
            '''
        ),
        (
            'human',
            f'''
            Assess whether the script can be completed using general knowledge:
            {script}
            '''
        )
    ]

    return llm.invoke(messages).content