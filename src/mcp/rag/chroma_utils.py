# mcp/rag/chroma_utils.py
# Este arquivo contém funções para interagir com o armazenamento vetorial Chroma,
# incluindo indexação e exclusão de documentos.

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
import logging

# ADICIONADO: Importar para carregar credenciais da nova config
from mcp.config import get_credentials # Importação corrigida

# Configura o logging para este módulo.
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Obtém as credenciais usando a nova função de configuração
credentials = get_credentials()

# Inicializa o embedding model do Google Generative AI (Gemini).
# O modelo padrão "models/embedding-001" é genérico para embeddings.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", credentials=credentials)

# Define o diretório onde o ChromaDB persistirá os dados.
CHROMA_PATH = "chroma_data"

# Inicializa o armazenamento vetorial Chroma com o diretório e o modelo de embeddings.
# Isso garante que o Chroma crie/carregue seu banco de dados neste diretório.
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """
    Carrega um documento, divide-o em pedaços e os indexa no Chroma.

    Args:
        file_path (str): O caminho para o arquivo a ser indexado.
        file_id (int): O ID do arquivo a ser associado aos pedaços de documento.

    Returns:
        bool: True se a indexação for bem-sucedida, False caso contrário.
    """
    try:
        # Carrega o documento baseado na extensão do arquivo.
        loader = None
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".html":
            loader = UnstructuredHTMLLoader(file_path)
        else:
            logging.error(f"Tipo de arquivo não suportado para indexação: {file_path}")
            print(f"Tipo de arquivo não suportado: {file_path}")
            return False

        documents = loader.load()
        logging.info(f"Documento {file_path} carregado. Número de páginas/documentos: {len(documents)}")

        # Adiciona o metadata file_id a cada documento antes de dividir
        for doc in documents:
            doc.metadata["file_id"] = file_id

        # Divide os documentos em pedaços para indexação.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits: List[Document] = text_splitter.split_documents(documents)
        logging.info(f"Documento {file_path} dividido em {len(splits)} pedaços.")

        vectorstore.add_documents(splits)
        logging.info(f"Documento do arquivo {file_path} (ID: {file_id}) indexado com sucesso no Chroma.")
        return True
    except Exception as e:
        logging.error(f"Erro ao indexar documento {file_path} (ID: {file_id}) no Chroma: {e}", exc_info=True)
        print(f"Erro ao indexar documento: {e}")
        return False


def delete_doc_from_chroma(file_id: int) -> bool:
    """
    Exclui todos os pedaços de documento associados a um determinado file_id do Chroma.

    Args:
        file_id (int): O ID do arquivo cujos pedaços devem ser excluídos.

    Returns:
        bool: True se a exclusão for bem-sucedida, False caso contrário.
    """
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        logging.info(f"Encontrados {len(docs.get('ids', []))} pedaços de documento para file_id {file_id} no Chroma.")
        print(f"Encontrados {len(docs.get('ids', []))} pedaços de documento para file_id {file_id}")

        vectorstore._collection.delete(where={"file_id": file_id})
        logging.info(f"Todos os documentos com file_id {file_id} excluídos do Chroma.")
        print(f"Todos os documentos com file_id {file_id} excluídos do Chroma.")
        return True
    except Exception as e:
        logging.error(f"Erro ao excluir documento com file_id {file_id} do Chroma: {e}", exc_info=True)
        print(f"Erro ao excluir documento do Chroma: {e}")
        return False