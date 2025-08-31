from typing import Dict, Any
import time
from langchain_core.messages import AIMessage

from ..logger import baml_logger, ExecutionTracker
from ..utils import (
    sanitize_for_logging, 
    extract_user_message_content, 
    create_error_context
)

def baml_tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    BAML Analysis node that directly calls BAML functions for structured text analysis.
    
    Uses BAML framework to perform:
    - Sentiment analysis with confidence scores
    - Key point extraction with importance levels  
    - Text summarization
    - Person information extraction (when relevant)
    """
    
    with ExecutionTracker(baml_logger, state):
        messages = state.get("messages", [])
        
        if not messages:
            baml_logger.warning("No messages in state")
            return {"messages": messages}
        
        # Extract user content to analyze
        user_content = extract_user_message_content(messages)
        
        if not user_content.strip():
            baml_logger.warning("Empty user content")
            return {"messages": messages}
        
        baml_logger.info(
            "Starting BAML analysis",
            text_length=len(user_content),
            text_preview=sanitize_for_logging(user_content)
        )
        
        try:
            # Import BAML client (generated from schemas)
            from baml_client import b
            
            analysis_results = {}
            
            # 1. Text Analysis - Always perform this
            start_time = time.time()
            baml_logger.info("Calling BAML AnalyzeText function")
            
            text_analysis = b.AnalyzeText(user_content)
            analysis_time = time.time() - start_time
            
            analysis_results["text_analysis"] = {
                "sentiment": text_analysis.sentiment.label,
                "confidence": text_analysis.sentiment.confidence,
                "key_points": [{"point": kp.point, "importance": kp.importance} for kp in text_analysis.key_points],
                "summary": text_analysis.summary,
                "word_count": text_analysis.word_count,
                "execution_time_ms": round(analysis_time * 1000, 2)
            }
            
            baml_logger.info(
                "Text analysis completed",
                sentiment=text_analysis.sentiment.label,
                confidence=text_analysis.sentiment.confidence,
                execution_time_ms=round(analysis_time * 1000, 2)
            )
            
            # 2. Person Extraction - Only if text contains person indicators
            if _should_extract_people(user_content):
                start_time = time.time()
                baml_logger.info("Calling BAML ExtractPersonInfo function")
                
                people = b.ExtractPersonInfo(user_content)
                extraction_time = time.time() - start_time
                
                if people:
                    analysis_results["people"] = [
                        {
                            "name": p.name,
                            "age": p.age,
                            "occupation": p.occupation,
                            "location": p.location
                        } for p in people
                    ]
                    
                    baml_logger.info(
                        "Person extraction completed",
                        people_found=len(people),
                        execution_time_ms=round(extraction_time * 1000, 2)
                    )
            
            # Create formatted response
            response_content = _format_analysis_results(analysis_results)
            
            # Add analysis message to conversation
            analysis_message = AIMessage(content=response_content)
            new_messages = messages + [analysis_message]
            
            baml_logger.info("BAML analysis completed successfully")
            
            return {
                "messages": new_messages,
                "analysis_complete": True,
                "baml_results": analysis_results
            }
            
        except ImportError:
            # BAML client not generated yet
            error_msg = """
The BAML client hasn't been generated yet. To enable structured analysis:

1. Install dependencies: `pip install -r requirements.txt`
2. Generate BAML client: `cd langgraph && baml-cli generate`
3. Restart the server: `langgraph dev`

I can still provide basic responses without structured analysis.
"""
            baml_logger.error("BAML client not available - run 'baml-cli generate'")
            
            error_message = AIMessage(content=error_msg.strip())
            return {
                "messages": messages + [error_message],
                "analysis_complete": True,
                "baml_setup_required": True
            }
            
        except Exception as e:
            error_context = create_error_context(
                e, 
                state, 
                {"text_length": len(user_content)}
            )
            
            baml_logger.log_error_with_context(e, error_context)
            
            error_message = AIMessage(
                content=f"Analysis error: {str(e)}. Please try again."
            )
            
            return {
                "messages": messages + [error_message],
                "analysis_complete": True,
                "error": str(e)
            }

def _should_extract_people(text: str) -> bool:
    """Simple heuristic to determine if person extraction is worthwhile"""
    import re
    
    # Look for person indicators
    person_words = ["person", "people", "name", "age", "work", "job", "employee", "manager"]
    has_person_words = any(word in text.lower() for word in person_words)
    
    # Look for potential names (capitalized words)
    has_names = bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text))
    
    return has_person_words or has_names

def _format_analysis_results(results: Dict[str, Any]) -> str:
    """Format BAML analysis results into readable response"""
    
    parts = ["## Text Analysis\n"]
    
    # Text analysis section
    if "text_analysis" in results:
        ta = results["text_analysis"]
        
        sentiment = ta["sentiment"].title()
        confidence = ta["confidence"]
        parts.append(f"**Sentiment**: {sentiment} (confidence: {confidence:.2f})")
        parts.append(f"**Word Count**: {ta['word_count']}")
        
        if ta["summary"]:
            parts.append(f"**Summary**: {ta['summary']}")
        
        # Key points
        if ta["key_points"]:
            parts.append("\n**Key Points**:")
            for i, kp in enumerate(ta["key_points"], 1):
                icon = {"high": "🔥", "medium": "⚡", "low": "💡"}.get(kp["importance"], "💡")
                parts.append(f"{i}. {icon} {kp['point']}")
    
    # People section
    if "people" in results and results["people"]:
        parts.append("\n## People Mentioned\n")
        
        for i, person in enumerate(results["people"], 1):
            details = [f"**{person['name']}**"]
            
            if person["age"]:
                details.append(f"Age: {person['age']}")
            if person["occupation"]:
                details.append(f"Occupation: {person['occupation']}")
            if person["location"]:
                details.append(f"Location: {person['location']}")
            
            parts.append(f"{i}. " + " | ".join(details))
    
    return "\n".join(parts)