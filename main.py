from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

CAMINHO_DB = "db"

prompt_template = """
Responda a pergunta do usuário:
{pergunta}

com base nessas informações:

{base_conhecimento}

Se você não souber a resposta, diga "Desculpe, não sei a resposta para essa pergunta." e não tente inventar uma resposta.
"""

pergunta = input("Digite sua pergunta: ")

#carregar o banco de dados
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)

#comparar a pergunta do usuario (embedding) com o meu banco de dados
resultado = db.similarity_search_with_relevance_scores(pergunta, k=3)
print(resultado[0])
print(len(resultado))