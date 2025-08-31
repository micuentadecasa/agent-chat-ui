import logging
import structlog
import sys
import time
from typing import Any, Dict, Optional, Callable
from functools import wraps
from pathlib import Path
import uuid

def configure_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_file: str = "langgraph.log",
    json_format: bool = False
) -> None:
    """Configure structured logging for the LangGraph application"""
    
    # Create logs directory if it doesn't exist
    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        # JSON format for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Human-readable format for development
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Add file handler if needed
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        logging.getLogger().addHandler(file_handler)

class NodeLogger:
    """Specialized logger for LangGraph nodes with tracing capabilities"""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.logger = structlog.get_logger(node_name)
        
    def info(self, message: str, **kwargs):
        """Log info message with node context"""
        self.logger.info(message, node=self.node_name, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message with node context"""
        self.logger.error(message, node=self.node_name, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with node context"""
        self.logger.warning(message, node=self.node_name, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message with node context"""
        self.logger.debug(message, node=self.node_name, **kwargs)
    
    def log_node_entry(self, state: Dict[str, Any], **kwargs):
        """Log entry into a node with state information"""
        message_count = len(state.get("messages", []))
        self.info(
            "Node execution started",
            message_count=message_count,
            state_keys=list(state.keys()),
            **kwargs
        )
    
    def log_node_exit(self, state: Dict[str, Any], execution_time: float, **kwargs):
        """Log exit from a node with results"""
        message_count = len(state.get("messages", []))
        self.info(
            "Node execution completed",
            message_count=message_count,
            execution_time_ms=round(execution_time * 1000, 2),
            state_keys=list(state.keys()),
            **kwargs
        )
    
    def log_llm_call(self, model_config: Dict[str, Any], input_tokens: Optional[int] = None):
        """Log LLM API call details"""
        self.info(
            "LLM API call initiated",
            model=model_config.get("model", "unknown"),
            base_url=model_config.get("base_url", "unknown"),
            temperature=model_config.get("temperature"),
            max_tokens=model_config.get("max_tokens"),
            input_tokens=input_tokens
        )
    
    def log_llm_response(self, response_time: float, output_tokens: Optional[int] = None, **kwargs):
        """Log LLM response details"""
        self.info(
            "LLM API call completed",
            response_time_ms=round(response_time * 1000, 2),
            output_tokens=output_tokens,
            **kwargs
        )
    
    def log_baml_call(self, function_name: str, input_data: Any):
        """Log BAML function call"""
        self.info(
            "BAML function call initiated",
            function=function_name,
            input_type=type(input_data).__name__,
            input_length=len(str(input_data)) if input_data else 0
        )
    
    def log_baml_result(self, function_name: str, success: bool, execution_time: float, **kwargs):
        """Log BAML function result"""
        self.info(
            "BAML function call completed",
            function=function_name,
            success=success,
            execution_time_ms=round(execution_time * 1000, 2),
            **kwargs
        )
    
    def log_routing_decision(self, next_node: str, reason: str, **kwargs):
        """Log routing decision to next node"""
        self.info(
            "Routing decision made",
            next_node=next_node,
            reason=reason,
            **kwargs
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context"""
        self.error(
            "Node execution error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            exc_info=True
        )

def timing_logger(logger: NodeLogger):
    """Decorator to log execution time of functions"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            try:
                logger.debug(f"Function {function_name} started")
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.debug(
                    f"Function {function_name} completed",
                    execution_time_ms=round(execution_time * 1000, 2)
                )
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Function {function_name} failed",
                    execution_time_ms=round(execution_time * 1000, 2),
                    error=str(e),
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator

class ExecutionTracker:
    """Context manager for tracking node execution"""
    
    def __init__(self, logger: NodeLogger, state: Dict[str, Any]):
        self.logger = logger
        self.state = state
        self.start_time = None
        self.execution_id = str(uuid.uuid4())[:8]
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log_node_entry(
            self.state,
            execution_id=self.execution_id
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if exc_type is not None:
            self.logger.log_error_with_context(
                exc_val,
                {
                    "execution_id": self.execution_id,
                    "execution_time": execution_time
                }
            )
        else:
            self.logger.log_node_exit(
                self.state,
                execution_time,
                execution_id=self.execution_id
            )

# Pre-configured loggers for each node
reasoning_logger = NodeLogger("reasoning_node")
baml_logger = NodeLogger("baml_tool_node")
graph_logger = NodeLogger("graph")
server_logger = NodeLogger("server")