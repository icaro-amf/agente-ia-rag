# 🤖 Agente de IA com RAG (Retrieval-Augmented Generation)

Este projeto é um sistema de Perguntas e Respostas (Q&A) baseado em documentos PDF locais. Ele utiliza a arquitetura RAG para ler arquivos PDF, vetorizar seu conteúdo e fornecer respostas precisas utilizando a inteligência artificial do **Google Gemini**.

> **Nota:** Este projeto foi inicialmente inspirado em uma videoaula do canal [Hashtag Programação](https://www.youtube.com/watch?v=0M8iO5ykY-E), mas foi **totalmente refatorado** para substituir a API paga da OpenAI pelas ferramentas gratuitas do ecossistema Google AI Studio (Gemini).

## 🚀 Tecnologias Utilizadas

*   **Linguagem:** Python
*   **Orquestração:** LangChain
*   **Banco de Dados Vetorial:** ChromaDB
*   **Embeddings:** `gemini-embedding-001` (Google Generative AI)
*   **LLM (Geração de Resposta):** `gemini-3.1-flash-lite` (Google Generative AI)
*   **Processamento de PDF:** PyPDFDirectoryLoader

## 📁 Estrutura do Projeto

*   `base/`: Pasta onde você deve colocar os arquivos `.pdf` que deseja consultar.
*   `criar_db.py`: Script responsável por ler os PDFs da pasta `base`, dividi-los em pedaços (chunks) e criar o banco de dados vetorial local.
*   `main.py`: Script principal de chat. Ele recebe a pergunta do usuário, busca no banco de dados e aciona o LLM para responder.
*   `requirements.txt`: Lista de dependências do Python.
*   `.env`: Arquivo (que você deve criar) para armazenar sua chave de API de forma segura.

## 🛠️ Como Instalar e Rodar

### 1. Clonar o repositório e instalar dependências
Certifique-se de ter o Python instalado. É recomendado criar um ambiente virtual (`.venv`).
```bash
# Clone o repositório
git clone [https://github.com/icaro-amf/agente-ia-rag.git](https://github.com/icaro-amf/agente-ia-rag.git)
cd agente-ia-rag

# Instale os pacotes necessários
pip install -r requirements.txt