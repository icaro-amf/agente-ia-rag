from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
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
def perguntar():
    pergunta = input("Digite sua pergunta: ")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)
    
    if (len(resultados) == 0 or resultados[0][1] < 0.6):
        print("Com base nas informações disponíveis, não encontrei uma resposta relevante para essa pergunta.")
        return
    
    textos_resultados = []
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultados.append(texto)

    base_conhecimento = "\n\n------\n\n".join(textos_resultados)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({
        "pergunta": pergunta,
        "base_conhecimento": base_conhecimento
    })

    modelo = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.3)
    texto_resposta = modelo.invoke(prompt).content

    if isinstance(texto_resposta, list):
        texto_resposta = texto_resposta[0]['text']
    else:
        texto_resposta = texto_resposta # Caso já venha como texto simples
        
    print("Resposta do modelo: ", texto_resposta)

perguntar()