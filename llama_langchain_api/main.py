import os
import streamlit as st
from langchain.llms import Replicate
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "r8_GILgHy3Lw888LR4uncdZVggiBw119Dq3wIVmN"

# Embeddings storage folder path
DB_FAISS_PATH = 'dbstorage/db_faiss'

# Load and preprocess the PDF document
loader = TextLoader('./text_data/divvy.txt')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use HuggingFace embeddings for transforming text into numerical vectors

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

# Set up the vector database
index_name = "llama-2-dd"
db = FAISS.load_local(DB_FAISS_PATH, embeddings)

# Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.25, "max_length": 3000}
)

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=False
)

chat_history = []
question = st.text_input("Enter your query:")
if st.button("Submit"):
    result = qa_chain({'question': question, 'chat_history': chat_history})
    st.write(result["answer"])