import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

PASTA_BASE = "base"

def criar_db():
    docs = carregar_docs()
    chunks = dividir_documentos(docs)
    vetorizar_chunks(chunks)

def carregar_docs():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    docs = carregador.load()
    return docs

def dividir_documentos(docs):
    separador_docs = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_docs.split_documents(docs)
    return chunks

def vetorizar_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2")

    db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="db")

    print("Base de dados criada com sucesso!")

criar_db()