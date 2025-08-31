import asyncio
import json
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..logger import reasoning_logger
from ..utils import sanitize_for_logging, create_error_context

class Context7SearchInput(BaseModel):
    """Input schema for Context7 search tool"""
    query: str = Field(..., description="Search query to find relevant information")
    max_results: int = Field(default=5, description="Maximum number of results to return")

class Context7Tool(BaseTool):
    """
    Tool for retrieving information using Context7 MCP server.
    
    This tool connects to a Context7 MCP server to search for and retrieve
    relevant information based on user queries.
    """
    
    name: str = "search_context7"
    description: str = (
        "Search for relevant information using Context7 MCP server. "
        "Use this tool when you need to find external information, "
        "research topics, or get additional context about subjects mentioned by the user."
    )
    args_schema: type[BaseModel] = Context7SearchInput
    
    def __init__(self, mcp_server_uri: str = "stdio://context7-server", **kwargs):
        super().__init__(**kwargs)
        self.mcp_server_uri = mcp_server_uri
        self._client = None
    
    async def _get_mcp_client(self):
        """Get or create MCP client connection"""
        if self._client is None:
            try:
                # Import MCP client (this will be available after installing mcp package)
                from mcp import ClientSession, StdioServerParameters
                import subprocess
                
                # Create server parameters
                server_params = StdioServerParameters(
                    command="context7-server",  # Adjust command as needed
                    args=[]
                )
                
                # Create client session
                self._client = ClientSession(server_params)
                
                reasoning_logger.info("MCP client connected to Context7 server")
                
            except ImportError:
                reasoning_logger.error("MCP package not available - install with: pip install mcp")
                raise ImportError("MCP package required for Context7 integration")
            except Exception as e:
                reasoning_logger.error(f"Failed to connect to Context7 MCP server: {e}")
                self._client = None
                raise
        
        return self._client
    
    def _run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Execute Context7 search synchronously"""
        try:
            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._arun(query, max_results))
                return result
            finally:
                loop.close()
        except Exception as e:
            reasoning_logger.error(f"Context7 search failed: {e}")
            return {
                "error": f"Search failed: {str(e)}",
                "error_type": type(e).__name__,
                "query": query
            }
    
    async def _arun(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Execute Context7 search asynchronously"""
        
        query_preview = sanitize_for_logging(query)
        
        reasoning_logger.info(
            "Context7 search initiated",
            query_preview=query_preview,
            max_results=max_results
        )
        
        try:
            client = await self._get_mcp_client()
            
            if client is None:
                return {
                    "error": "MCP client not available",
                    "message": "Could not connect to Context7 server"
                }
            
            # Initialize connection if needed
            async with client:
                # List available tools from the MCP server
                tools_result = await client.list_tools()
                
                reasoning_logger.debug(
                    "Available MCP tools",
                    tool_count=len(tools_result.tools) if tools_result.tools else 0
                )
                
                # Look for search-related tools
                search_tool = None
                for tool in tools_result.tools or []:
                    if 'search' in tool.name.lower() or 'query' in tool.name.lower():
                        search_tool = tool
                        break
                
                if search_tool is None:
                    # Try common tool names
                    common_search_names = ["search", "query", "find", "get_info", "retrieve"]
                    for tool_name in common_search_names:
                        try:
                            # Try to call the tool
                            search_result = await client.call_tool(
                                tool_name,
                                {
                                    "query": query,
                                    "limit": max_results
                                }
                            )
                            search_tool = {"name": tool_name}
                            break
                        except:
                            continue
                
                if search_tool is None:
                    available_tools = [tool.name for tool in tools_result.tools or []]
                    reasoning_logger.warning(
                        "No search tool found in Context7 server",
                        available_tools=available_tools
                    )
                    return {
                        "error": "No search tool available",
                        "available_tools": available_tools,
                        "suggestion": "Check Context7 server configuration"
                    }
                
                # Execute the search
                reasoning_logger.info(f"Executing search with tool: {search_tool['name'] if isinstance(search_tool, dict) else search_tool.name}")
                
                tool_name = search_tool['name'] if isinstance(search_tool, dict) else search_tool.name
                
                search_result = await client.call_tool(
                    tool_name,
                    {
                        "query": query,
                        "max_results": max_results,
                        "limit": max_results  # Alternative parameter name
                    }
                )
                
                # Process results
                results = []
                if hasattr(search_result, 'content') and search_result.content:
                    for content_item in search_result.content:
                        if hasattr(content_item, 'text'):
                            try:
                                # Try to parse as JSON if possible
                                result_data = json.loads(content_item.text)
                                if isinstance(result_data, list):
                                    results.extend(result_data)
                                else:
                                    results.append(result_data)
                            except json.JSONDecodeError:
                                # If not JSON, treat as plain text
                                results.append({
                                    "content": content_item.text,
                                    "type": "text"
                                })
                
                reasoning_logger.info(
                    "Context7 search completed",
                    results_found=len(results),
                    query=query_preview
                )
                
                return {
                    "results": results,
                    "query": query,
                    "count": len(results),
                    "source": "context7"
                }
                
        except Exception as e:
            error_context = create_error_context(
                e,
                {"query": query, "max_results": max_results},
                {"mcp_server_uri": self.mcp_server_uri}
            )
            
            reasoning_logger.log_error_with_context(e, error_context)
            
            return {
                "error": f"Context7 search failed: {str(e)}",
                "error_type": type(e).__name__,
                "query": query
            }

# Fallback implementation for when MCP is not available
class MockContext7Tool(BaseTool):
    """
    Mock Context7 tool for development/testing when MCP server is not available
    """
    
    name: str = "search_context7"
    description: str = (
        "Search for relevant information (Mock implementation - MCP server not available). "
        "This is a fallback that provides basic responses when Context7 is not configured."
    )
    args_schema: type[BaseModel] = Context7SearchInput
    
    def _run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Mock search implementation"""
        
        reasoning_logger.warning(
            "Using mock Context7 tool - MCP server not available",
            query=sanitize_for_logging(query)
        )
        
        # Provide a helpful mock response
        mock_results = [
            {
                "title": "Mock Result for: " + query[:50],
                "content": f"This is a mock response for the query '{query}'. To get real results, please configure the Context7 MCP server.",
                "type": "mock",
                "source": "mock_context7"
            }
        ]
        
        return {
            "results": mock_results,
            "query": query,
            "count": len(mock_results),
            "source": "mock_context7",
            "warning": "Mock implementation - Context7 MCP server not configured"
        }

def create_context7_tool(mcp_server_uri: Optional[str] = None, use_mock: bool = False) -> BaseTool:
    """
    Factory function to create Context7 tool with proper error handling
    
    Args:
        mcp_server_uri: URI for the MCP server
        use_mock: Force use of mock implementation
    """
    
    if use_mock:
        reasoning_logger.info("Creating mock Context7 tool")
        return MockContext7Tool()
    
    try:
        # Try to import MCP to check if it's available
        import mcp
        reasoning_logger.info("Creating real Context7 tool with MCP")
        return Context7Tool(mcp_server_uri=mcp_server_uri or "stdio://context7-server")
    
    except ImportError:
        reasoning_logger.warning("MCP package not available, using mock Context7 tool")
        return MockContext7Tool()

# Default tool instance
context7_tool = create_context7_tool()