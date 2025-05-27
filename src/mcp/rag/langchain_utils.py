# mcp/rag/langchain_utils.py
# Este arquivo implementa o núcleo do sistema RAG usando LangChain,
# configurando o modelo de linguagem, retriever e a cadeia RAG.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
import logging
# Importação corrigida para o vectorstore
from mcp.rag.chroma_utils import vectorstore

# Importação corrigida para as credenciais
from mcp.config import get_credentials

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Obtém as credenciais usando a nova função de configuração
credentials = get_credentials()

# Cria um retriever a partir do vectorstore. Isso permite buscar documentos relevantes.
retriever = vectorstore.as_retriever()

# Template de prompt para contextualizar a pergunta do usuário com o histórico de chat.
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Dado um histórico de chat e uma pergunta de acompanhamento, gere uma pergunta autônoma que possa ser usada para recuperar documentos relevantes. Se não houver histórico de chat, retorne a pergunta original."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Template de prompt para a cadeia de perguntas e respostas.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de IA útil. Use o seguinte contexto para responder à pergunta do usuário. Se você não souber a resposta, diga que não sabe. Não tente inventar uma resposta."),
    ("system", "Contexto: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model: str = "gemini-2.0-flash"):
    """
    Cria e retorna a cadeia RAG (Retrieval-Augmented Generation).

    Args:
        model (str): O nome do modelo de linguagem a ser usado (padrão: "gemini-2.0-flash").

    Returns:
        Runnable: A cadeia RAG completa pronta para ser invocada.
    """
    # Inicializa o modelo de linguagem de chat do Google Generative AI (Gemini).
    # Passando as credenciais explicitamente.
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.7, credentials=credentials)

    # Cria um retriever ciente do histórico.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Cria uma cadeia de documentos que combina os documentos recuperados com a pergunta
    # para gerar uma resposta usando o LLM e o qa_prompt.
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Cria a cadeia de recuperação que orquestra o retriever ciente do histórico
    # e a cadeia de perguntas e respostas.
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    return rag_chain