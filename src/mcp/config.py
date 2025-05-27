# mcp/config.py
import os
import logging
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_cached_credentials = None # Cache para evitar carregar as credenciais múltiplas vezes

def get_credentials() -> Credentials:
    """
    Carrega e retorna as credenciais da conta de serviço para a API do Google.
    As credenciais são armazenadas em cache após a primeira carga bem-sucedida.
    """
    global _cached_credentials
    if _cached_credentials:
        return _cached_credentials

    # O caminho para o arquivo credentials.json (assumindo que está na raiz do projeto)
    # Acessamos a raiz do projeto (D:\dev\mcp_server\) a partir de mcp/config.py
    CREDENTIALS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "credentials.json")

    if not os.path.exists(CREDENTIALS_FILE):
        logging.critical(f"Erro: Arquivo de credenciais não encontrado em {CREDENTIALS_FILE}. A aplicação não pode continuar.")
        raise FileNotFoundError(f"Arquivo de credenciais não encontrado: {CREDENTIALS_FILE}")

    try:
        _cached_credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
        logging.info(f"Credenciais carregadas com sucesso de: {CREDENTIALS_FILE}")
        return _cached_credentials
    except Exception as e:
        logging.critical(f"Erro ao carregar credenciais de {CREDENTIALS_FILE}: {e}", exc_info=True)
        raise RuntimeError(f"Falha ao carregar credenciais da conta de serviço: {e}")

# Exemplo de configuração de modelos (será expandido no mcp/router.py)
MODEL_CONFIGS = {
    "gemini-2.0-flash": {
        "engine": "gemini",
        "description": "Google Gemini 2.0 Flash model.",
        "max_tokens": 8192,
        "cost_per_token_input": 0.0000001, # Exemplo de custo por token
        "cost_per_token_output": 0.0000002,
    },
    # Adicione outros modelos aqui
    # "gpt-4o": {
    #     "engine": "openai",
    #     "description": "OpenAI GPT-4o model.",
    #     "max_tokens": 128000,
    #     "cost_per_token_input": 0.000005,
    #     "cost_per_token_output": 0.000015,
    # },
}