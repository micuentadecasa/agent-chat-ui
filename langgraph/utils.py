import re
from typing import List, Union, Any
from langchain_core.messages import BaseMessage


def estimate_tokens(messages: Union[List[BaseMessage], BaseMessage, str, List[str]]) -> int:
    """
    Estimate token count for messages or text.
    
    This is a rough approximation based on the rule of thumb:
    - 1 token ≈ 4 characters for English text
    - Adjusted for message structure overhead
    """
    
    if isinstance(messages, str):
        return len(messages) // 4
    
    if isinstance(messages, list) and len(messages) > 0:
        if isinstance(messages[0], str):
            # List of strings
            total_chars = sum(len(msg) for msg in messages)
            return total_chars // 4
        
        elif hasattr(messages[0], 'content'):
            # List of message objects
            total_chars = 0
            for msg in messages:
                content = getattr(msg, 'content', '')
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            total_chars += len(item['text'])
                        elif isinstance(item, str):
                            total_chars += len(item)
            
            # Add overhead for message structure (role, metadata, etc.)
            overhead_per_message = 10  # rough estimate
            return (total_chars // 4) + (len(messages) * overhead_per_message)
    
    elif hasattr(messages, 'content'):
        # Single message object
        content = getattr(messages, 'content', '')
        if isinstance(content, str):
            return len(content) // 4 + 10  # +10 for message overhead
        elif isinstance(content, list):
            # Multimodal content
            total_chars = 0
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    total_chars += len(item['text'])
                elif isinstance(item, str):
                    total_chars += len(item)
            return total_chars // 4 + 10
    
    return 0


def sanitize_for_logging(text: str, max_length: int = 200) -> str:
    """
    Sanitize text for logging by removing sensitive information and truncating.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Remove potential sensitive patterns (basic patterns)
    sensitive_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card pattern
        r'\b[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[:=]\s*\S+',  # API key pattern
        r'\b[Tt][Oo][Kk][Ee][Nn]\s*[:=]\s*\S+',  # Token pattern
    ]
    
    sanitized = text
    for pattern in sensitive_patterns:
        sanitized = re.sub(pattern, '[REDACTED]', sanitized)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized


def extract_user_message_content(messages: List[BaseMessage]) -> str:
    """Extract the content from the last user message in a conversation."""
    if not messages:
        return ""
    
    # Find the last message from a human/user
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            return getattr(msg, 'content', '')
        elif 'human' in str(type(msg)).lower() or 'user' in str(type(msg)).lower():
            return getattr(msg, 'content', '')
    
    # Fallback to the last message content
    last_message = messages[-1]
    return getattr(last_message, 'content', '')


def format_execution_summary(state: dict, execution_time: float) -> dict:
    """Create a summary of node execution for logging."""
    messages = state.get('messages', [])
    
    summary = {
        'total_messages': len(messages),
        'execution_time_ms': round(execution_time * 1000, 2),
        'state_keys': list(state.keys()),
    }
    
    # Add message type breakdown
    message_types = {}
    for msg in messages:
        msg_type = getattr(msg, 'type', 'unknown')
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    
    summary['message_types'] = message_types
    
    # Add content length stats
    if messages:
        content_lengths = []
        for msg in messages:
            content = getattr(msg, 'content', '')
            if isinstance(content, str):
                content_lengths.append(len(content))
            elif isinstance(content, list):
                # Handle multimodal content
                total_len = 0
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        total_len += len(item['text'])
                    elif isinstance(item, str):
                        total_len += len(item)
                content_lengths.append(total_len)
        
        if content_lengths:
            summary['content_stats'] = {
                'avg_length': sum(content_lengths) // len(content_lengths),
                'total_length': sum(content_lengths),
                'max_length': max(content_lengths),
                'min_length': min(content_lengths)
            }
    
    return summary


def validate_state(state: dict) -> tuple[bool, str]:
    """Validate that the state has the required structure."""
    if not isinstance(state, dict):
        return False, "State must be a dictionary"
    
    if 'messages' not in state:
        return False, "State must contain 'messages' key"
    
    messages = state['messages']
    if not isinstance(messages, list):
        return False, "Messages must be a list"
    
    # Validate message structure
    for i, msg in enumerate(messages):
        if not hasattr(msg, 'content'):
            return False, f"Message {i} missing 'content' attribute"
    
    return True, "State is valid"


def create_error_context(error: Exception, state: dict, additional_info: dict = None) -> dict:
    """Create comprehensive error context for logging."""
    context = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'state_valid': validate_state(state)[0],
        'state_keys': list(state.keys()) if isinstance(state, dict) else 'invalid',
    }
    
    if isinstance(state, dict):
        messages = state.get('messages', [])
        context['message_count'] = len(messages)
        
        if messages:
            last_message = messages[-1]
            context['last_message_type'] = getattr(last_message, 'type', 'unknown')
            content = getattr(last_message, 'content', '')
            context['last_message_length'] = len(str(content))
    
    if additional_info:
        context['additional_info'] = additional_info
    
    return context