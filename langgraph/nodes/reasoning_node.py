from typing import Dict, Any, List
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ..config import config
from ..logger import reasoning_logger, ExecutionTracker, timing_logger
from ..utils import estimate_tokens, sanitize_for_logging, extract_user_message_content, format_execution_summary, create_error_context

@timing_logger(reasoning_logger)
def create_reasoning_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for LM Studio"""
    llm_config = config.get_llm_config()
    reasoning_logger.debug("Creating LLM instance", config=llm_config)
    return ChatOpenAI(**llm_config)

def reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reasoning node that handles general reasoning and thinking tasks.
    
    This node takes the user's input and provides thoughtful analysis,
    reasoning, or answers to questions before passing to specialized tools.
    """
    
    with ExecutionTracker(reasoning_logger, state) as tracker:
        messages = state.get("messages", [])
        
        if not messages:
            reasoning_logger.warning("No messages in state, returning empty")
            return {"messages": messages}
        
        # Extract user message for analysis
        user_content = extract_user_message_content(messages)
        user_content_sanitized = sanitize_for_logging(user_content)
        
        reasoning_logger.info(
            "Processing user message",
            message_length=len(user_content),
            message_preview=user_content_sanitized,
            total_messages=len(messages)
        )
        
        # Create the reasoning LLM
        llm_config = config.get_llm_config()
        input_tokens = estimate_tokens(messages)
        reasoning_logger.log_llm_call(llm_config, input_tokens=input_tokens)
        
        try:
            llm = create_reasoning_llm()
            
            # System prompt for reasoning
            system_prompt = SystemMessage(content="""
You are a helpful AI assistant focused on clear reasoning and analysis. 
Your role is to:
1. Think through problems step by step
2. Provide clear explanations for your reasoning
3. Ask clarifying questions when needed
4. Identify when specialized analysis tools might be helpful

Be concise but thorough in your responses. If the user's input contains text that would benefit from detailed analysis (sentiment analysis, information extraction, etc.), mention that you can perform deeper analysis using specialized tools.
""")
            
            # Create the message sequence
            chat_messages = [system_prompt] + messages
            
            # Generate response with timing
            start_time = time.time()
            response = llm.invoke(chat_messages)
            response_time = time.time() - start_time
            
            # Log response details
            response_content = getattr(response, 'content', '')
            output_tokens = estimate_tokens([response])
            
            reasoning_logger.log_llm_response(
                response_time,
                output_tokens=output_tokens,
                response_length=len(response_content)
            )
            
            # Add the AI response to messages
            new_messages = messages + [response]
            
            # Create execution summary
            execution_summary = format_execution_summary(
                {"messages": new_messages}, 
                response_time
            )
            
            # Update state with tracking info
            updated_state = {
                "messages": new_messages,
                "reasoning_complete": True,
                "next": "baml_tool_node",  # Indicate next node to process
                "reasoning_time": response_time,
                "reasoning_tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                }
            }
            
            reasoning_logger.info(
                "Reasoning completed successfully",
                response_preview=sanitize_for_logging(response_content),
                **execution_summary
            )
            
            return updated_state
            
        except Exception as e:
            error_context = create_error_context(
                e, 
                state, 
                {
                    "llm_config": llm_config,
                    "input_tokens": input_tokens,
                    "user_content_length": len(user_content)
                }
            )
            
            reasoning_logger.log_error_with_context(e, error_context)
            
            # Handle errors gracefully
            error_message = AIMessage(
                content=f"I encountered an error while reasoning: {str(e)}. Please try again."
            )
            
            return {
                "messages": messages + [error_message],
                "reasoning_complete": True,
                "error": str(e)
            }

def should_continue_to_baml(state: Dict[str, Any]) -> str:
    """
    Determine if we should continue to the BAML tool node.
    This function decides the routing logic.
    """
    
    # Check if there's an error
    if state.get("error"):
        reasoning_logger.log_routing_decision("end", "Error detected in state")
        return "end"
    
    # Check if reasoning is complete
    if state.get("reasoning_complete"):
        # Look for keywords that suggest text analysis would be helpful
        messages = state.get("messages", [])
        if messages:
            user_content = extract_user_message_content(messages)
            user_content_lower = user_content.lower()
            
            analysis_keywords = [
                "analyze", "sentiment", "extract", "summary", 
                "key points", "information", "data", "text analysis",
                "people", "person", "names", "extraction"
            ]
            
            # Check for analysis keywords
            keyword_match = any(keyword in user_content_lower for keyword in analysis_keywords)
            
            # Check for long text that might benefit from analysis
            is_long_text = len(user_content) > 200
            
            if keyword_match or is_long_text:
                reasoning_logger.log_routing_decision(
                    "baml_tool_node", 
                    "Analysis keywords found or long text detected",
                    keyword_match=keyword_match,
                    is_long_text=is_long_text,
                    text_length=len(user_content),
                    matched_keywords=[kw for kw in analysis_keywords if kw in user_content_lower]
                )
                return "baml_tool_node"
        
        # Otherwise, end the conversation
        reasoning_logger.log_routing_decision("end", "No analysis needed, ending conversation")
        return "end"
    
    reasoning_logger.log_routing_decision("end", "Reasoning not complete")
    return "end"