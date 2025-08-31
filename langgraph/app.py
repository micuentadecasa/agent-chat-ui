#!/usr/bin/env python3

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add the langgraph directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import graph, create_compiled_graph, run_graph, stream_graph
from config import config
from logger import server_logger, configure_logging

# Initialize logging
configure_logging(**config.get_logging_config())

# Health check for LM Studio connectivity
async def check_lm_studio_health() -> bool:
    """Check if LM Studio is accessible"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{config.LM_STUDIO_BASE_URL.rstrip('/v1')}/v1/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
    except Exception as e:
        server_logger.warning(f"LM Studio health check failed: {e}")
        return False

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    server_logger.info("Starting LangGraph server")
    server_logger.info(f"Server configuration: {config.SERVER_HOST}:{config.SERVER_PORT}")
    server_logger.info(f"LM Studio URL: {config.LM_STUDIO_BASE_URL}")
    
    # Check LM Studio connectivity on startup
    lm_studio_ok = await check_lm_studio_health()
    if lm_studio_ok:
        server_logger.info("✓ LM Studio connectivity verified")
    else:
        server_logger.warning("⚠ LM Studio not accessible - please ensure it's running")
    
    # Initialize graph if not already done
    if graph is None:
        server_logger.error("Failed to initialize graph - server may not function properly")
    else:
        server_logger.info("✓ LangGraph workflow initialized")
    
    yield
    
    server_logger.info("Shutting down LangGraph server")

# Create FastAPI app
app = FastAPI(
    title="LangGraph Agent Server",
    description="LangGraph backend with reasoning and BAML analysis nodes",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation persistence")
    stream: bool = Field(False, description="Whether to stream the response")

class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    reasoning_time: Optional[float] = None
    analysis_time: Optional[float] = None
    total_tokens: Optional[int] = None
    thread_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    lm_studio_connected: bool
    graph_initialized: bool
    version: str = "1.0.0"

# Utility functions
def convert_to_langchain_messages(messages: List[ChatMessage]) -> List:
    """Convert Pydantic messages to LangChain messages"""
    langchain_messages = []
    
    for msg in messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            langchain_messages.append(SystemMessage(content=msg.content))
    
    return langchain_messages

def convert_from_langchain_messages(messages: List) -> List[ChatMessage]:
    """Convert LangChain messages to Pydantic messages"""
    chat_messages = []
    
    for msg in messages:
        if hasattr(msg, 'type') and hasattr(msg, 'content'):
            role_mapping = {
                'human': 'user',
                'ai': 'assistant',
                'system': 'system'
            }
            role = role_mapping.get(msg.type, 'assistant')
            chat_messages.append(ChatMessage(role=role, content=msg.content))
    
    return chat_messages

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    lm_studio_ok = await check_lm_studio_health()
    graph_ok = graph is not None
    
    status = "healthy" if (lm_studio_ok and graph_ok) else "degraded"
    
    server_logger.info(
        "Health check requested",
        status=status,
        lm_studio_connected=lm_studio_ok,
        graph_initialized=graph_ok
    )
    
    return HealthResponse(
        status=status,
        lm_studio_connected=lm_studio_ok,
        graph_initialized=graph_ok
    )

@app.get("/info")
async def get_info():
    """Get server information - compatible with LangGraph deployment format"""
    return {
        "title": "LangGraph Agent Server",
        "description": "LangGraph backend with reasoning and BAML analysis nodes",
        "version": "1.0.0",
        "config": {
            "lm_studio_url": config.LM_STUDIO_BASE_URL,
            "server_port": config.SERVER_PORT,
            "logging_enabled": config.LOG_TO_FILE
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint - non-streaming"""
    
    if graph is None:
        server_logger.error("Graph not initialized")
        raise HTTPException(status_code=500, detail="Graph not initialized")
    
    server_logger.info(
        "Chat request received",
        message_count=len(request.messages),
        thread_id=request.thread_id,
        stream=request.stream
    )
    
    try:
        # Convert messages
        langchain_messages = convert_to_langchain_messages(request.messages)
        
        # Run the graph
        final_state = run_graph(graph, langchain_messages, request.thread_id)
        
        # Convert response
        response_messages = convert_from_langchain_messages(final_state.get("messages", []))
        
        # Calculate token usage
        reasoning_tokens = final_state.get("reasoning_tokens", {})
        total_tokens = reasoning_tokens.get("total", 0)
        
        response = ChatResponse(
            messages=response_messages,
            reasoning_time=final_state.get("reasoning_time"),
            analysis_time=final_state.get("baml_analysis_time"),
            total_tokens=total_tokens,
            thread_id=request.thread_id
        )
        
        server_logger.info(
            "Chat request completed",
            response_message_count=len(response_messages),
            total_tokens=total_tokens,
            reasoning_time=response.reasoning_time,
            analysis_time=response.analysis_time
        )
        
        return response
        
    except Exception as e:
        server_logger.error(
            "Chat request failed",
            error=str(e),
            error_type=type(e).__name__,
            thread_id=request.thread_id,
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint"""
    
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    
    server_logger.info(
        "Streaming chat request received",
        message_count=len(request.messages),
        thread_id=request.thread_id
    )
    
    async def generate_stream():
        try:
            # Convert messages
            langchain_messages = convert_to_langchain_messages(request.messages)
            
            # Stream the graph execution
            for state in stream_graph(graph, langchain_messages, request.thread_id):
                # Convert and yield state updates
                if isinstance(state, dict) and "messages" in state:
                    messages = convert_from_langchain_messages(state["messages"])
                    
                    # Create streaming response chunk
                    chunk = {
                        "messages": [msg.dict() for msg in messages],
                        "reasoning_complete": state.get("reasoning_complete", False),
                        "analysis_complete": state.get("analysis_complete", False),
                        "thread_id": request.thread_id
                    }
                    
                    yield f"data: {chunk}\n\n"
            
            # Send final completion marker
            yield "data: [DONE]\n\n"
            
            server_logger.info("Streaming chat completed", thread_id=request.thread_id)
            
        except Exception as e:
            server_logger.error(
                "Streaming chat failed",
                error=str(e),
                thread_id=request.thread_id,
                exc_info=True
            )
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    server_logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )

# Development server runner
def run_server():
    """Run the development server"""
    
    print(f"Starting LangGraph server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"LM Studio URL: {config.LM_STUDIO_BASE_URL}")
    print(f"Logging level: {config.LOG_LEVEL}")
    print(f"Log file: {config.LOG_FILE if config.LOG_TO_FILE else 'Console only'}")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.DEBUG_MODE,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )

if __name__ == "__main__":
    run_server()