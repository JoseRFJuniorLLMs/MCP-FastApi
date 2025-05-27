# mcp_server/main.py
from fastapi import FastAPI
from mcp_server.router_api import router as mcp_api_router
import logging

# Configura o logging para a aplicação.
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="MCP AI Server",
    description="Multi-Channel Processor AI Server with Smart Routing and RAG support.",
    version="0.0.1",
)

# Inclui o APIRouter do router_api.py
app.include_router(mcp_api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the MCP AI Server! Access /api/v1/docs for API documentation."}