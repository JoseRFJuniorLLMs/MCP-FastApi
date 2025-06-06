Sistema RAG com FastAPI e LangChain (Google Gemini)
Este projeto é uma aplicação web robusta construída com FastAPI que implementa um sistema de Geração Aumentada de Recuperação (RAG). Ele utiliza o framework LangChain para orquestrar a interação com modelos de linguagem Google Gemini e um banco de dados vetorial ChromaDB para armazenamento e recuperação de documentos. Para persistência de logs de chat e metadados de documentos, um banco de dados SQLite é utilizado.

🌟 Funcionalidades
Chat com RAG: Interaja com o modelo Google Gemini (especificamente gemini-1.5-flash), que utiliza documentos previamente indexados para fornecer respostas contextualmente relevantes.

Upload e Indexação de Documentos: Carregue documentos nos formatos PDF, DOCX e HTML. Estes documentos são automaticamente divididos em pedaços e indexados no ChromaDB, tornando-os pesquisáveis pelo sistema RAG.

Listagem de Documentos: Visualize uma lista de todos os documentos que foram carregados e indexados na aplicação, juntamente com seus IDs e carimbos de data/hora de upload.

Exclusão de Documentos: Remova documentos indexados do sistema. A exclusão remove o documento tanto do banco de dados SQLite (metadados) quanto do ChromaDB (embeddings).

Persistência de Chat: O histórico de conversas com o chatbot é armazenado em um banco de dados SQLite, permitindo que o modelo mantenha o contexto em interações futuras.

🛠️ Tecnologias Utilizadas
FastAPI: Framework web moderno, rápido (alto desempenho) para construir APIs com Python.

LangChain: Framework para desenvolvimento de aplicações baseadas em LLMs, facilitando a orquestração de componentes de IA.

Google Gemini (via langchain-google-genai): Modelos de linguagem para geração de respostas (gemini-1.5-flash) e embeddings (models/text-embedding-004).

ChromaDB: Banco de dados vetorial leve e embutido para armazenamento de embeddings de documentos.

SQLite: Banco de dados relacional simples e sem servidor para persistência de logs de chat e metadados de documentos.

Uvicorn: Servidor ASGI (Asynchronous Server Gateway Interface) para rodar a aplicação FastAPI.

google-auth, google-oauth2-credentials: Bibliotecas Python para autenticação com as APIs do Google, utilizando credenciais de conta de serviço.

python-multipart: Biblioteca para lidar com uploads de arquivos em requisições HTTP.

🚀 Configuração e Instalação
Siga os passos abaixo para configurar e rodar o projeto localmente.

Pré-requisitos
Python 3.10+: Recomenda-se Python 3.10, 3.11 ou 3.12 para melhor compatibilidade com as bibliotecas.

Conta Google Cloud: Com a API Google Gemini habilitada e uma Conta de Serviço configurada.

1. Clonar o Repositório
Primeiro, clone o repositório para sua máquina local e navegue até a pasta do projeto:

git clone https://github.com/JoseRFJuniorLLMs/FastApi-LangChain.git
cd FastApi-LangChain

2. Criar e Ativar o Ambiente Virtual
É uma prática recomendada usar um ambiente virtual para isolar as dependências do projeto:

python -m venv .venv
# Para Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Para macOS/Linux
source .venv/bin/activate

3. Instalar Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias. Crie um arquivo requirements.txt na raiz do seu projeto com o seguinte conteúdo:

requirements.txt:

fastapi
uvicorn
langchain
langchain-core
langchain-community
langchain-google-genai
chromadb
pypdf
docx2txt
unstructured
python-multipart
google-auth
google-auth-oauthlib
protobuf==3.20.3 # Versão específica para evitar TypeError com protobuf

Depois, instale as dependências:

pip install -r requirements.txt

4. Configurar Credenciais do Google Cloud
Este projeto utiliza uma Conta de Serviço para autenticação com as APIs do Google.

No seu projeto Google Cloud, navegue até "IAM & Admin" -> "Service Accounts".

Crie uma nova conta de serviço ou selecione uma existente.

Vá para a aba "Keys" (Chaves) da sua conta de serviço.

Clique em "Add Key" -> "Create new key".

Selecione "JSON" como o tipo de chave e clique em "Create".

Um arquivo JSON será baixado para sua máquina. Renomeie-o para credentials.json.

Coloque este arquivo credentials.json na raiz do seu projeto (na mesma pasta onde está o src e o README.md).

ATENÇÃO: O arquivo credentials.json contém informações sensíveis. Ele NÃO DEVE ser versionado no Git. Este projeto já possui um arquivo .gitignore configurado para excluí-lo.

5. Configurar Variável de Ambiente para Credenciais
Para que a aplicação encontre suas credenciais, defina a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS no seu terminal antes de rodar a aplicação:

# Para Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS="D:\dev\FastApi-LangChain\credentials.json"
# Para macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS="/caminho/completo/para/seu/projeto/credentials.json"

Certifique-se de que o caminho D:\dev\FastApi-LangChain\credentials.json (ou o equivalente no Linux/macOS) esteja correto para a localização do seu arquivo.

▶️ Rodando a Aplicação
Com o ambiente virtual ativado, as dependências instaladas e as credenciais configuradas, você pode iniciar o servidor Uvicorn:

# Certifique-se de que está na pasta raiz do projeto (FastApi-LangChain)
uvicorn src.main:app --reload

Você verá mensagens como:

INFO:      Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:      Started reloader process [...] using WatchFiles
INFO:      Started server process [...]
INFO:      Waiting for application startup.
INFO:      Application startup complete.

🌐 Acessando a API
Após a aplicação iniciar, você pode acessar a documentação interativa da API no seu navegador:

Swagger UI (Documentação Interativa): http://127.0.0.1:8000/docs

ReDoc (Documentação Alternativa): http://127.0.0.1:8000/redoc

Use a interface do Swagger UI (/docs) para testar os endpoints diretamente.

⚙️ Endpoints da API
A aplicação expõe os seguintes endpoints:

POST /chat: Envia uma pergunta ao modelo Gemini. A resposta será aumentada com base nos documentos indexados no ChromaDB.

Request Body (Exemplo):

{
  "question": "Qual é o principal tópico do documento X?",
  "session_id": "minha_sessao_123",
  "model": "gemini-1.5-flash"
}

POST /upload-doc: Carrega um documento (PDF, DOCX, HTML) para ser processado, dividido e indexado no ChromaDB.

Request Body: multipart/form-data com o campo file.

GET /list-docs: Lista todos os documentos que foram carregados e seus IDs.

POST /delete-doc: Exclui um documento indexado e seus metadados.

Request Body (Exemplo):

{
  "file_id": 1
}

📂 Estrutura do Projeto
.: Pasta raiz do projeto.

src/: Contém os módulos da aplicação principal.

src/main.py: O arquivo principal da aplicação FastAPI, onde os endpoints são definidos e a lógica é orquestrada.

src/langchain_utils.py: Contém a lógica para construir a cadeia RAG (Retrieval-Augmented Generation) usando LangChain e Google Gemini.

src/chroma_utils.py: Lida com a interação com o ChromaDB para indexação e recuperação de documentos.

src/db_utils.py: Gerencia o banco de dados SQLite para logs de chat e metadados de documentos.

credentials.json: Seu arquivo de credenciais da conta de serviço do Google Cloud (MANTENHA-O FORA DO GIT!).

.gitignore: Define os arquivos e pastas a serem ignorados pelo Git (incluindo credentials.json, .venv/ e chroma_db/).

requirements.txt: Lista as dependências do Python.

rag_app.db: (Gerado automaticamente) O arquivo do banco de dados SQLite para logs e metadatos.

chroma_db/: (Gerado automaticamente) O diretório de persistência para o ChromaDB.

⚠️ Solução de Problemas Comuns
DefaultCredentialsError: Certifique-se de que seu arquivo credentials.json está na raiz do projeto e que ele é um arquivo de credenciais de Conta de Serviço válido. Verifique se a variável de ambiente GOOGLE_APPLICATION_CREDENTIALS está apontando para o caminho correto do arquivo.

Not Found na URL raiz (/): Isso é esperado. Acesse http://127.0.0.1:8000/docs para ver a documentação da API.

TypeError: Descriptors cannot be created directly: Este erro é geralmente resolvido fazendo o downgrade do pacote protobuf para a versão 3.20.3 (conforme especificado no requirements.txt).

ModuleNotFoundError ou ImportError: Verifique cuidadosamente se todos os arquivos estão com as importações corretas, especialmente as importações relativas (from .module import ...) dentro da pasta src.# MCP-FastApi
# MCP-FastApi
