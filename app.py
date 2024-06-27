
from gc import callbacks
import chainlit as cl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
import os
import getpass
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

#1. Set path for AirBnB 10K pdf document & load OpenAI API Key & embedding model
PROJECT_DIR = Path(__file__).parent
SOURCE_PDF_DIR = PROJECT_DIR / 'data' / 'AirBnB.pdf'

load_dotenv()
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

#Embeddings using Hugging face interface endpoint
embedding_model = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)


#2. Load PDF Document
loader = PyMuPDFLoader(SOURCE_PDF_DIR)
documents = loader.load()


#3. Perform chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
documents = text_splitter.split_documents(documents)


#4 Store embeddings in QDrant vector store in memory
""" from langchain_community.vectorstores import Qdrant
qdrant_vector_store = Qdrant.from_documents(
    documents,
    embedding_model,
    location=":memory:",
    collection_name="AirBnB 10K Document",
)
qdrant_retriever = qdrant_vector_store.as_retriever() """

if os.path.exists("./app"):
    vectorstore = FAISS.load_local(
        "./app", 
        embedding_model, 
        allow_dangerous_deserialization=True # this is necessary to load the vectorstore from disk as it's stored as a `.pkl` file.
    )
    hf_retriever = vectorstore.as_retriever()
    print("Loaded Vectorstore")
else:
    print("Indexing Files")
    os.makedirs("./app", exist_ok=True)
    for i in range(0, len(documents), 32):
        if i == 0:
            vectorstore = FAISS.from_documents(documents[i:i+32], embedding_model)
            continue
        vectorstore.add_documents(documents[i:i+32])
    vectorstore.save_local("./app")

qdrant_retriever = vectorstore.as_retriever()

#5 Quey for search
query = "What is the 'maximum number of shares to be sold under the 10b5-1 Trading plan' by Brian Chesky?"

#6 Setting up RAG Prompt Template
from langchain_core.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

#8 Create LLM endpoint
openai_chat_model = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    temperature=0.3,
    repetition_penalty=1.15,
    huggingfacehub_api_token=HF_TOKEN,
)

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough


#9 Integrate with chainlit
@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "AirBnB 10K Q&A Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """
    lcel_rag_chain = (
        {"context": itemgetter("query") | qdrant_retriever, "query": itemgetter("query")}
        | rag_prompt | openai_chat_model | StrOutputParser()
    )
    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)
    print("++++++++++++++++++++++++++++++++++++++++")
    print("LCEL chain set")
    print("++++++++++++++++++++++++++++++++++++++++")

    
    
@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")
    print("Query : " + message.content)

    for chunk in await cl.make_async(lcel_rag_chain.stream)(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()