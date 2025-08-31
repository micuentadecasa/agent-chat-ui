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

3. **BAML Functions** (`baml/`)
   - Direct BAML function calls for text analysis
   - Person information extraction
   - Structured output with type safety

4. **Graph Workflow** (`graph.py`)
   - StateGraph connecting reasoning → analysis
   - Memory persistence with checkpointing
   - Compatible with langgraph dev server

## Configuration

Environment variables are loaded from `langgraph/.env`:

```env
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
LM_STUDIO_MODEL=local-model

# LangGraph Development Configuration
LANGGRAPH_DEV_PORT=2024

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

4. Start the development server:
```bash
cd langgraph && langgraph dev
```

The server will start on http://localhost:2024 with:
- API endpoints for the Agent Chat UI
- Built-in streaming support
- Hot reload during development
- LangGraph Studio integration

## Logging

Comprehensive structured logging with:
- Node execution tracking
- LLM call monitoring
- Tool execution logging  
- Performance metrics
- Error context

Logs are written to both console and file with configurable levels.

## Integration with Agent Chat UI

The LangGraph development server is fully compatible with the Agent Chat UI:
- Serves on port 2024 by default (configurable via LANGGRAPH_DEV_PORT)
- Built-in LangGraph deployment API format
- Native streaming support
- Thread persistence with memory checkpointing
- Hot reload for rapid development