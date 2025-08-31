from typing import Dict, Any, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes.reasoning_node import reasoning_node, should_continue_to_baml
from .nodes.baml_tool_node import baml_tool_node
from .config import config
from .logger import graph_logger, configure_logging

# Initialize logging
configure_logging(**config.get_logging_config())

class GraphState(TypedDict):
    """
    State schema for the LangGraph workflow.
    
    Represents the data passed between nodes in the graph.
    """
    messages: List[BaseMessage]
    reasoning_complete: bool
    analysis_complete: bool
    reasoning_time: float
    baml_analysis_time: float
    reasoning_tokens: Dict[str, int]
    baml_analysis: Dict[str, Any]
    baml_functions_called: List[str]
    error: str
    baml_setup_required: bool

def create_graph() -> StateGraph:
    """
    Create and configure the LangGraph workflow.
    
    The graph consists of:
    1. reasoning_node - Handles general reasoning and analysis
    2. baml_tool_node - Performs structured text analysis using BAML
    
    Returns:
        StateGraph: Configured graph ready for compilation
    """
    
    graph_logger.info("Creating LangGraph workflow")
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("reasoning_node", reasoning_node)
    workflow.add_node("baml_tool_node", baml_tool_node)
    
    # Set entry point
    workflow.set_entry_point("reasoning_node")
    
    # Add conditional edge from reasoning to baml or end
    workflow.add_conditional_edges(
        "reasoning_node",
        should_continue_to_baml,
        {
            "baml_tool_node": "baml_tool_node",
            "end": END
        }
    )
    
    # Add edge from baml_tool_node to end
    workflow.add_edge("baml_tool_node", END)
    
    graph_logger.info("Graph structure created successfully")
    graph_logger.debug(
        "Graph configuration",
        entry_point="reasoning_node",
        nodes=["reasoning_node", "baml_tool_node"],
        edges=["reasoning_node -> baml_tool_node/END", "baml_tool_node -> END"]
    )
    
    return workflow

def create_compiled_graph(with_memory: bool = True):
    """
    Create and compile the graph with optional memory persistence.
    
    Args:
        with_memory: Whether to use memory checkpointing for conversation persistence
        
    Returns:
        Compiled graph ready for execution
    """
    
    graph_logger.info(f"Compiling graph with memory: {with_memory}")
    
    try:
        workflow = create_graph()
        
        # Add memory saver if requested
        checkpointer = MemorySaver() if with_memory else None
        
        if checkpointer:
            graph_logger.info("Memory checkpointer enabled")
        
        # Compile the graph
        compiled_graph = workflow.compile(checkpointer=checkpointer)
        
        graph_logger.info("Graph compiled successfully")
        
        return compiled_graph
        
    except Exception as e:
        graph_logger.error(
            "Failed to compile graph",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise

def run_graph(compiled_graph, messages: List[BaseMessage], thread_id: str = None) -> Dict[str, Any]:
    """
    Execute the graph with the given messages.
    
    Args:
        compiled_graph: Compiled LangGraph instance
        messages: List of messages to process
        thread_id: Optional thread ID for conversation persistence
        
    Returns:
        Final state after graph execution
    """
    
    input_state = {"messages": messages}
    config_dict = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    
    graph_logger.info(
        "Starting graph execution",
        message_count=len(messages),
        thread_id=thread_id,
        has_thread=thread_id is not None
    )
    
    try:
        # Execute the graph
        final_state = compiled_graph.invoke(input_state, config=config_dict)
        
        # Log execution summary
        execution_summary = {
            "final_message_count": len(final_state.get("messages", [])),
            "reasoning_completed": final_state.get("reasoning_complete", False),
            "analysis_completed": final_state.get("analysis_complete", False),
            "total_reasoning_time": final_state.get("reasoning_time", 0),
            "total_analysis_time": final_state.get("baml_analysis_time", 0),
            "functions_called": final_state.get("baml_functions_called", []),
            "has_error": bool(final_state.get("error")),
            "baml_setup_required": final_state.get("baml_setup_required", False)
        }
        
        graph_logger.info(
            "Graph execution completed",
            thread_id=thread_id,
            **execution_summary
        )
        
        return final_state
        
    except Exception as e:
        graph_logger.error(
            "Graph execution failed",
            error=str(e),
            error_type=type(e).__name__,
            thread_id=thread_id,
            input_message_count=len(messages),
            exc_info=True
        )
        raise

def stream_graph(compiled_graph, messages: List[BaseMessage], thread_id: str = None):
    """
    Stream the graph execution, yielding intermediate states.
    
    Args:
        compiled_graph: Compiled LangGraph instance
        messages: List of messages to process  
        thread_id: Optional thread ID for conversation persistence
        
    Yields:
        Intermediate states during graph execution
    """
    
    input_state = {"messages": messages}
    config_dict = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    
    graph_logger.info(
        "Starting graph streaming",
        message_count=len(messages),
        thread_id=thread_id
    )
    
    try:
        for state in compiled_graph.stream(input_state, config=config_dict):
            graph_logger.debug(
                "Graph state update",
                state_keys=list(state.keys()) if isinstance(state, dict) else "unknown",
                thread_id=thread_id
            )
            yield state
            
        graph_logger.info("Graph streaming completed", thread_id=thread_id)
        
    except Exception as e:
        graph_logger.error(
            "Graph streaming failed",
            error=str(e),
            error_type=type(e).__name__,
            thread_id=thread_id,
            exc_info=True
        )
        raise

# Create the graph instance for langgraph dev
graph = create_compiled_graph(with_memory=True)
graph_logger.info("Graph instance created for langgraph dev")