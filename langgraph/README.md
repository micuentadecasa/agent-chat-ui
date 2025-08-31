# LangGraph Backend with BAML Analysis

A Python-based LangGraph application that provides an AI reasoning pipeline with structured text analysis using the BAML framework.

## Architecture

### Core Components

1. **Reasoning Node** (`nodes/reasoning_node.py`)
   - Handles general reasoning and thinking tasks
   - Uses local LLM through LM Studio
   - Determines routing to analysis tools

2. **BAML Tool Node** (`nodes/baml_tool_node.py`)  
   - Uses BAML-based tools for structured analysis
   - Text sentiment analysis with key points
   - Person information extraction
   - LLM agent with tool calling pattern

3. **BAML Tools** (`tools/baml_tools.py`)
   - `BAMLTextAnalysisTool` - Sentiment, summary, key points
   - `BAMLPersonExtractionTool` - Extract person details
   - Proper LangChain tool integration

4. **Graph Workflow** (`graph.py`)
   - StateGraph connecting reasoning → analysis
   - Memory persistence with checkpointing
   - Streaming and batch execution modes

5. **FastAPI Server** (`app.py`)
   - REST API endpoints compatible with Agent Chat UI
   - Health checks and error handling
   - Streaming and non-streaming chat endpoints

## Configuration

Environment variables are loaded from `langgraph/.env`:

```env
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=local-model

# Server Configuration  
SERVER_HOST=0.0.0.0
SERVER_PORT=2024

# Logging Configuration
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_FILE=logs/langgraph.log
```

## BAML Setup

BAML schemas are organized by node:

- `baml/main.baml` - Shared configuration and data models
- `baml/baml_tool_node.baml` - Text analysis functions with tests

To generate BAML client:
```bash
cd langgraph
baml-cli generate --from ./baml/main.baml
```

## Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure LM Studio:
   - Start LM Studio with a model loaded
   - Ensure server is running on localhost:1234

3. Generate BAML client:
```bash
cd langgraph && baml-cli generate
```

4. Run the server:
```bash
cd langgraph && python app.py
```

## API Endpoints

- `GET /health` - Health check
- `GET /info` - Server information  
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - Streaming chat

## Logging

Comprehensive structured logging with:
- Node execution tracking
- LLM call monitoring
- Tool execution logging  
- Performance metrics
- Error context

Logs are written to both console and file with configurable levels.

## Integration with Agent Chat UI

The server is compatible with the Agent Chat UI frontend:
- Serves on port 2024 by default
- Implements LangGraph deployment API format
- Supports both streaming and batch modes
- Handles thread persistence for conversations