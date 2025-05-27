# mcp/pydantic_models.py
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# Enum para definir os nomes dos modelos de linguagem permitidos.
class ModelName(str, Enum):
    # Definimos os modelos que você quer usar.
    # Por enquanto, apenas o Gemini Flash, mas você pode adicionar outros aqui.
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    # Adicione outros modelos conforme necessário, por exemplo:
    # GPT_4O = "gpt-4o"
    # CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    # LOCAL_VLLM = "local-vllm" # Para um modelo rodando localmente via vLLM

# Modelo para a entrada de uma consulta de chat.
class QueryInput(BaseModel):
    question: str  # A pergunta do usuário (obrigatória).
    session_id: str = Field(default=None)  # ID da sessão (opcional, será gerado se não for fornecido).
    model: ModelName = Field(default=ModelName.GEMINI_2_0_FLASH)  # Modelo de linguagem a ser usado, com padrão.
    context_type: str = Field(default="chat", description="Tipo de contexto da requisição: 'chat' ou 'rag'.") # NOVO: Para roteamento

# Modelo para a resposta de uma consulta de chat.
class QueryResponse(BaseModel):
    answer: str  # A resposta gerada pelo modelo.
    session_id: str  # O ID da sessão.
    model: ModelName  # O modelo usado para gerar a resposta.

# Modelo para informações sobre um documento indexado.
class DocumentInfo(BaseModel):
    id: int  # Identificador único do documento.
    filename: str  # Nome do arquivo do documento.
    upload_timestamp: datetime  # Carimbo de data/hora do upload.

# Modelo para a requisição de exclusão de arquivo.
class DeleteFileRequest(BaseModel):
    file_id: int  # ID do arquivo a ser excluído.