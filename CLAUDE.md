# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

The project uses pnpm as the package manager:

- `pnpm install` - Install dependencies
- `pnpm dev` - Start development server on http://localhost:3000
- `pnpm build` - Build production version
- `pnpm start` - Start production server
- `pnpm lint` - Run ESLint
- `pnpm lint:fix` - Fix ESLint issues automatically
- `pnpm format` - Format code with Prettier
- `pnpm format:check` - Check code formatting

## Architecture Overview

This is a Next.js application that enables chatting with LangGraph servers through a web interface. The app serves as a frontend for AI agents built with LangGraph.

### Core Components

- **Stream Provider** (`src/providers/Stream.tsx`) - Main context provider managing LangGraph connections using `@langchain/langgraph-sdk/react`. Handles authentication, thread management, and real-time streaming
- **Thread Component** (`src/components/thread/`) - Core chat interface with message rendering, file uploads, and agent interaction handling
- **API Passthrough** (`src/app/api/[..._path]/route.ts`) - Proxy layer using `langgraph-nextjs-api-passthrough` for production deployments

### Key Libraries

- **LangGraph SDK**: `@langchain/langgraph-sdk` and `@langchain/langgraph-sdk/react` for agent communication
- **UI Components**: Radix UI primitives with Tailwind CSS styling
- **State Management**: React Context with `nuqs` for URL state synchronization
- **File Handling**: Built-in file upload system for multimodal inputs

### Configuration

Environment variables control deployment behavior:
- `NEXT_PUBLIC_API_URL` - LangGraph server URL (bypasses setup form)
- `NEXT_PUBLIC_ASSISTANT_ID` - Graph/assistant ID to use
- `LANGGRAPH_API_URL` - Backend LangGraph URL for API passthrough
- `LANGSMITH_API_KEY` - Server-side authentication for production

### Message Handling

The app processes LangGraph message types:
- Messages with `id` starting with `do-not-render-` are filtered from UI
- Models tagged with `langsmith:nostream` prevent live streaming
- Custom UI messages handled via `uiMessageReducer` from the SDK

### Production Setup

Two deployment patterns:
1. **API Passthrough** - Uses proxy in `src/app/api/` with server-side auth
2. **Custom Authentication** - Direct client connection with custom LangGraph auth

The app automatically shows a setup form when required environment variables are missing, allowing dynamic configuration of deployment URL and assistant ID.

## LangGraph Backend

The `langgraph/` directory contains a complete Python backend implementation:

### Architecture
- **Two-node workflow**: Reasoning Node → BAML Tool Node
- **BAML Integration**: Structured text analysis using BAML framework tools
- **LM Studio Connection**: Local LLM via OpenAI-compatible API
- **Tool-based Design**: BAML functions implemented as LangChain tools
- **Comprehensive Logging**: Structured logging with performance tracking

### Key Commands
- `cd langgraph && python app.py` - Start LangGraph server on port 2024
- `baml-cli generate` - Generate BAML client from schema files
- `pip install -r requirements.txt` - Install Python dependencies

### Configuration
Environment settings in `langgraph/.env`:
- LM Studio connection (localhost:1234 by default)
- Server settings (host, port, logging)
- BAML environment configuration

The LangGraph backend provides structured AI analysis capabilities that complement the Next.js frontend, creating a complete agent chat system.