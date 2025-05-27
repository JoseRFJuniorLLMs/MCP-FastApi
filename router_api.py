# mcp_server/router_api.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from mcp.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest # Importação corrigida
from mcp.rag.langchain_utils import get_rag_chain # Importação corrigida
from mcp.rag.db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, \
    delete_document_record # Importação corrigida
from mcp.rag.chroma_utils import index_document_to_chroma, delete_doc_from_chroma # Importação corrigida
import os
import uuid
import logging
import shutil
import sys # Necessário para sys.modules

# O logging já está configurado em main.py, então podemos remover esta linha daqui se quisermos
# logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# **************** MUDANÇA CRUCIAL: De FastAPI() para APIRouter() ****************
router = APIRouter()

# Removido o print de debug do sys.modules, pois não é mais necessário aqui
# try:
#     import mcp.pydantic_models as pydantic_models_debug_module
#     print(f"Pydantic models imported from (debug): {pydantic_models_debug_module.__file__}")
#     if hasattr(pydantic_models_debug_module.ModelName, 'GEMINI_2_0_FLASH'):
#         print("ModelName.GEMINI_2_0_FLASH exists in pydantic_models_debug_module.")
#     else:
#         print("ModelName.GEMINI_2_0_FLASH does NOT exist in pydantic_models_debug_module.")
# except ImportError as e:
#     print(f"Could not import mcp.pydantic_models for debug: {e}")
# except AttributeError as e:
#     print(f"Attribute error during pydantic_models debug: {e}")
# **************** FIM DA MUDANÇA PARA DEBUG ****************


@router.post("/chat", response_model=QueryResponse) # Use router.post
async def chat(query_input: QueryInput):
    """
    Endpoint para interações de chat com o sistema RAG.

    Args:
        query_input (QueryInput): Objeto contendo a pergunta do usuário,
                                   ID da sessão (opcional) e o modelo a ser usado.

    Returns:
        QueryResponse: A resposta do modelo, o ID da sessão e o modelo usado.
    """
    # Se um session_id não for fornecido na requisição, gere um novo
    if query_input.session_id is None:
        session_id = str(uuid.uuid4())
        logging.info(f"Nova sessão iniciada com ID: {session_id}")
    else:
        session_id = query_input.session_id
        logging.info(f"Sessão existente ID: {session_id}")

    try:
        # Recupera o histórico de chat da sessão para o modelo
        chat_history = get_chat_history(session_id)

        # Obtém a cadeia RAG com o modelo especificado.
        # Agora o modelo vem do QueryInput
        rag_chain = get_rag_chain(query_input.model.value)

        # Invoca a cadeia RAG com a pergunta do usuário e o histórico de chat.
        response = rag_chain.invoke({"input": query_input.question, "chat_history": chat_history})

        # Insere a interação no log.
        insert_application_logs(session_id, query_input.question, response["answer"], query_input.model.value)
        logging.info(f"Resposta gerada para sessão {session_id}, modelo {query_input.model.value}.")

        # Retorna a resposta.
        return QueryResponse(answer=response["answer"], session_id=session_id, model=query_input.model)

    except Exception as e:
        logging.error(f"Erro no endpoint /chat para sessão {session_id}: {e}", exc_info=True)
        # Retorna um erro HTTP 500 em caso de exceção.
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno ao processar a requisição: {e}")


@router.post("/uploadfile/") # Use router.post
async def create_upload_file(file: UploadFile):
    """
    Endpoint para upload de arquivos (PDF, DOCX, HTML) e indexação no Chroma.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".pdf", ".docx", ".html"]:
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Por favor, envie .pdf, .docx ou .html.")

    # Caminho temporário para salvar o arquivo.
    # Certifique-se de que a pasta 'temp_docs' exista ou seja criada.
    upload_folder = "temp_docs"
    os.makedirs(upload_folder, exist_ok=True)
    temp_file_path = os.path.join(upload_folder, file.filename)

    try:
        # Salva o arquivo temporariamente.
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Arquivo {file.filename} salvo temporariamente em {temp_file_path}.")

        # Insere o registro do documento no banco de dados.
        file_id = insert_document_record(file.filename)
        logging.info(f"Registro do documento {file.filename} inserido no DB com ID {file_id}.")

        # Indexa o documento no Chroma.
        # Passa o file_id para que os chunks sejam associados a ele.
        index_success = index_document_to_chroma(temp_file_path, file_id)

        if index_success:
            logging.info(f"Arquivo {file.filename} (ID: {file_id}) processado e indexado com sucesso.")
            return {"message": f"Arquivo '{file.filename}' processado e indexado com sucesso!", "file_id": file_id}
        else:
            logging.error(f"Falha ao indexar o arquivo {file.filename} (ID: {file_id}).")
            # Se a indexação falhar, tentamos remover o registro do DB
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Falha ao indexar o arquivo '{file.filename}'.")
    except Exception as e:
        logging.error(f"Erro no endpoint /uploadfile para {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar o arquivo: {e}")
    finally:
        # Garante que o arquivo temporário seja removido.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Arquivo temporário {temp_file_path} removido.")


@router.get("/documents", response_model=List[DocumentInfo]) # Use router.get
async def list_documents():
    """
    Endpoint para listar todos os documentos atualmente indexados no sistema.
    """
    try:
        documents = get_all_documents()
        logging.info(f"Listados {len(documents)} documentos.")
        return documents
    except Exception as e:
        logging.error(f"Erro ao listar documentos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao listar documentos: {e}")


@router.delete("/documents", response_model=dict) # Use router.delete
async def delete_document(request: DeleteFileRequest):
    """
    Endpoint para excluir um documento do sistema e do armazenamento vetorial.
    """
    file_id = request.file_id
    logging.info(f"Tentativa de exclusão do documento com file_id: {file_id}")

    try:
        # Primeiro, tenta excluir do Chroma.
        chroma_delete_success = delete_doc_from_chroma(file_id)
        if chroma_delete_success:
            logging.info(f"Documento com file_id {file_id} excluído do Chroma.")
            # Se a exclusão do Chroma for bem-sucedida, tenta excluir do banco de dados.
            db_delete_success = delete_document_record(file_id)
            if db_delete_success:
                logging.info(f"Documento com file_id {file_id} excluído do banco de dados.")
                return {"message": f"Documento com file_id {file_id} excluído com sucesso do sistema."}
            else:
                logging.error(
                    f"Excluído do Chroma, mas falha ao excluir documento com file_id {file_id} do banco de dados.")
                # Retorna um erro 500 se a exclusão do DB falhar.
                raise HTTPException(status_code=500,
                                    detail=f"Excluído do Chroma, mas falha ao excluir documento com file_id {file_id} do banco de dados.")
        else:
            logging.error(f"Falha ao excluir documento com file_id {file_id} do Chroma.")
            # Retorna um erro 500 se a exclusão do Chroma falhar.
            raise HTTPException(status_code=500, detail=f"Falha ao excluir documento com file_id {file_id} do Chroma.")
    except Exception as e:
        logging.error(f"Erro geral ao excluir documento com file_id {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao excluir documento: {e}")