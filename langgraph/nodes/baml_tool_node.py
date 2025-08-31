from typing import Dict, Any, List
import time
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from ..config import config
from ..logger import baml_logger, ExecutionTracker, timing_logger
from ..utils import (
    sanitize_for_logging, 
    extract_user_message_content, 
    format_execution_summary, 
    create_error_context
)
from ..tools.baml_tools import get_baml_tools, should_use_person_extraction

@timing_logger(baml_logger)
def create_analysis_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for tool use"""
    llm_config = config.get_llm_config()
    baml_logger.debug("Creating analysis LLM with tools", config=llm_config)
    return ChatOpenAI(**llm_config)

def baml_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analysis node that uses BAML tools for structured text analysis.
    
    This node creates an LLM agent that can call BAML tools for:
    - Text sentiment analysis with key points and summary
    - Person information extraction
    """
    
    with ExecutionTracker(baml_logger, state) as tracker:
        messages = state.get("messages", [])
        
        if not messages:
            baml_logger.warning("No messages in state, returning empty")
            return {"messages": messages}
        
        # Extract text to analyze
        user_content = extract_user_message_content(messages)
        user_content_sanitized = sanitize_for_logging(user_content)
        
        baml_logger.info(
            "Starting tool-based analysis",
            text_length=len(user_content),
            text_preview=user_content_sanitized,
            total_messages=len(messages)
        )
        
        try:
            # Get BAML tools
            available_tools = get_baml_tools()
            
            # Create LLM with tools
            llm = create_analysis_llm()
            llm_with_tools = llm.bind_tools(available_tools)
            
            # Create system message for tool use
            from langchain_core.messages import SystemMessage
            
            system_message = SystemMessage(content=f"""
You are an AI assistant with access to specialized analysis tools. You have been asked to analyze the following text:

"{user_content[:500]}{'...' if len(user_content) > 500 else ''}"

Available tools:
1. text_analysis - For sentiment analysis, key points extraction, and text summarization
2. extract_people - For extracting information about people mentioned in the text

Instructions:
1. ALWAYS use the text_analysis tool to analyze the provided text
2. Use the extract_people tool if the text mentions specific people with details
3. Present the results in a clear, organized format
4. If a tool returns an error, explain what went wrong and suggest next steps

Please analyze this text using the appropriate tools.
""")
            
            # Create conversation with system message
            tool_messages = [system_message] + messages[-1:]  # Include only the latest user message
            
            start_time = time.time()
            
            baml_logger.info("Invoking LLM with BAML tools", tool_count=len(available_tools))
            
            # Get initial response (may include tool calls)
            response = llm_with_tools.invoke(tool_messages)
            
            # Handle tool calls if present
            tool_results = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                baml_logger.info(f"Processing {len(response.tool_calls)} tool calls")
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    
                    baml_logger.info(f"Executing tool: {tool_name}", args=tool_args)
                    
                    # Find and execute the tool
                    tool_instance = None
                    for tool in available_tools:
                        if tool.name == tool_name:
                            tool_instance = tool
                            break
                    
                    if tool_instance:
                        try:
                            tool_result = tool_instance.run(tool_args)
                            tool_results.append({
                                'tool': tool_name,
                                'result': tool_result,
                                'success': True
                            })
                            baml_logger.info(f"Tool {tool_name} executed successfully")
                        except Exception as e:
                            tool_results.append({
                                'tool': tool_name,
                                'error': str(e),
                                'success': False
                            })
                            baml_logger.error(f"Tool {tool_name} failed: {e}")
                    else:
                        baml_logger.error(f"Tool {tool_name} not found")
            
            # If we have tool results, create a follow-up response
            if tool_results:
                # Create a summary of tool results
                results_summary = create_tool_results_summary(tool_results)
                
                # Create follow-up message asking LLM to format the results
                follow_up_message = SystemMessage(content=f"""
Based on the tool execution results below, please provide a comprehensive analysis summary:

{results_summary}

Present this information in a clear, user-friendly format with appropriate sections and formatting.
""")
                
                final_response = llm.invoke([follow_up_message])
                response_content = getattr(final_response, 'content', str(final_response))
            else:
                response_content = getattr(response, 'content', str(response))
            
            total_time = time.time() - start_time
            
            # Create analysis message
            analysis_message = AIMessage(content=response_content)
            new_messages = messages + [analysis_message]
            
            # Create execution summary
            execution_summary = format_execution_summary(
                {"messages": new_messages}, 
                total_time
            )
            
            # Update state
            updated_state = {
                "messages": new_messages,
                "analysis_complete": True,
                "baml_analysis_time": total_time,
                "tools_used": [tr['tool'] for tr in tool_results if tr['success']],
                "tool_results": tool_results
            }
            
            baml_logger.info(
                "Tool-based analysis completed",
                tools_executed=len([tr for tr in tool_results if tr['success']]),
                total_execution_time=total_time,
                **execution_summary
            )
            
            return updated_state
            
        except Exception as e:
            error_context = create_error_context(
                e, 
                state, 
                {
                    "text_length": len(user_content),
                    "tools_available": len(get_baml_tools())
                }
            )
            
            baml_logger.log_error_with_context(e, error_context)
            
            # Handle errors gracefully
            error_message = AIMessage(
                content=f"I encountered an error during analysis: {str(e)}. Please try again."
            )
            
            return {
                "messages": messages + [error_message],
                "analysis_complete": True,
                "error": str(e)
            }

def create_tool_results_summary(tool_results: List[Dict[str, Any]]) -> str:
    """Create a formatted summary of tool execution results for LLM processing"""
    
    summary_parts = []
    
    for result in tool_results:
        tool_name = result.get('tool', 'Unknown Tool')
        
        if result.get('success', False):
            tool_data = result.get('result', {})
            
            summary_parts.append(f"## {tool_name.replace('_', ' ').title()} Results")
            
            if tool_name == 'text_analysis':
                # Handle text analysis results
                if 'error' not in tool_data:
                    sentiment = tool_data.get('sentiment', {})
                    summary_parts.append(f"- Sentiment: {sentiment.get('label', 'unknown').title()} (confidence: {sentiment.get('confidence', 0):.2f})")
                    summary_parts.append(f"- Word Count: {tool_data.get('word_count', 0)}")
                    
                    if tool_data.get('summary'):
                        summary_parts.append(f"- Summary: {tool_data['summary']}")
                    
                    key_points = tool_data.get('key_points', [])
                    if key_points:
                        summary_parts.append("- Key Points:")
                        for i, kp in enumerate(key_points, 1):
                            summary_parts.append(f"  {i}. [{kp.get('importance', 'medium')}] {kp.get('point', '')}")
                else:
                    summary_parts.append(f"- Error: {tool_data.get('error', 'Unknown error')}")
            
            elif tool_name == 'extract_people':
                # Handle person extraction results
                if 'error' not in tool_data:
                    people = tool_data.get('people', [])
                    summary_parts.append(f"- People Found: {len(people)}")
                    
                    for i, person in enumerate(people, 1):
                        person_info = [f"Name: {person.get('name', 'Unknown')}"]
                        if person.get('age'):
                            person_info.append(f"Age: {person['age']}")
                        if person.get('occupation'):
                            person_info.append(f"Occupation: {person['occupation']}")
                        if person.get('location'):
                            person_info.append(f"Location: {person['location']}")
                        
                        summary_parts.append(f"  {i}. {' | '.join(person_info)}")
                else:
                    summary_parts.append(f"- Error: {tool_data.get('error', 'Unknown error')}")
            
            summary_parts.append("")  # Add spacing
            
        else:
            # Handle tool execution errors
            error_msg = result.get('error', 'Unknown error occurred')
            summary_parts.append(f"## {tool_name.replace('_', ' ').title()} - FAILED")
            summary_parts.append(f"- Error: {error_msg}")
            summary_parts.append("")
    
    return "\n".join(summary_parts)