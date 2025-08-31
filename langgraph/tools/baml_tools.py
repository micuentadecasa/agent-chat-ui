from typing import Any, Dict, List, Optional
import time
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..logger import baml_logger
from ..utils import sanitize_for_logging, create_error_context

class TextAnalysisInput(BaseModel):
    """Input schema for text analysis tool"""
    text: str = Field(..., description="Text to analyze for sentiment, key points, and summary")

class PersonExtractionInput(BaseModel):
    """Input schema for person extraction tool"""
    text: str = Field(..., description="Text to extract person information from")

class BAMLTextAnalysisTool(BaseTool):
    """
    Tool for structured text analysis using BAML framework.
    
    This tool uses BAML to analyze text and extract:
    - Sentiment with confidence score
    - Key points with importance levels
    - Summary
    - Word count
    """
    
    name: str = "text_analysis"
    description: str = (
        "Analyze text to extract sentiment (positive/negative/neutral with confidence), "
        "key points with importance levels, summary, and word count. "
        "Use this tool when you need structured analysis of text content."
    )
    args_schema: type[BaseModel] = TextAnalysisInput
    
    def _run(self, text: str) -> Dict[str, Any]:
        """Execute text analysis using BAML"""
        
        baml_logger.log_baml_call("AnalyzeText", text)
        text_preview = sanitize_for_logging(text)
        
        baml_logger.info(
            "Text analysis tool called",
            text_length=len(text),
            text_preview=text_preview
        )
        
        try:
            # Import BAML client (generated from schema)
            from baml_client import b
            
            start_time = time.time()
            result = b.AnalyzeText(text)
            execution_time = time.time() - start_time
            
            # Convert BAML result to dictionary
            analysis_result = {
                "sentiment": {
                    "label": result.sentiment.label,
                    "confidence": result.sentiment.confidence
                },
                "key_points": [
                    {
                        "point": kp.point,
                        "importance": kp.importance
                    } for kp in result.key_points
                ],
                "summary": result.summary,
                "word_count": result.word_count,
                "analysis_time_ms": round(execution_time * 1000, 2)
            }
            
            baml_logger.log_baml_result(
                "AnalyzeText", 
                True, 
                execution_time,
                sentiment=result.sentiment.label,
                confidence=result.sentiment.confidence,
                key_points_count=len(result.key_points),
                word_count=result.word_count
            )
            
            return analysis_result
            
        except ImportError:
            error_msg = (
                "BAML client not available. Please run 'baml-cli generate' "
                "to generate the client from the BAML schema files."
            )
            baml_logger.error("BAML client import failed", error=error_msg)
            
            return {
                "error": "BAML client not available",
                "message": error_msg,
                "setup_required": True
            }
            
        except Exception as e:
            error_context = create_error_context(
                e, 
                {"text_length": len(text)}, 
                {"function": "AnalyzeText", "text_preview": text_preview}
            )
            
            baml_logger.log_error_with_context(e, error_context)
            
            return {
                "error": f"Text analysis failed: {str(e)}",
                "error_type": type(e).__name__
            }

class BAMLPersonExtractionTool(BaseTool):
    """
    Tool for extracting person information using BAML framework.
    
    This tool uses BAML to extract structured information about people mentioned in text:
    - Names (required)
    - Ages (if mentioned)
    - Occupations (if mentioned)
    - Locations (if mentioned)
    """
    
    name: str = "extract_people"
    description: str = (
        "Extract information about people mentioned in text including names, ages, "
        "occupations, and locations. Use this tool when you need to identify and "
        "extract structured information about people from text content."
    )
    args_schema: type[BaseModel] = PersonExtractionInput
    
    def _run(self, text: str) -> Dict[str, Any]:
        """Execute person extraction using BAML"""
        
        baml_logger.log_baml_call("ExtractPersonInfo", text)
        text_preview = sanitize_for_logging(text)
        
        baml_logger.info(
            "Person extraction tool called",
            text_length=len(text),
            text_preview=text_preview
        )
        
        try:
            # Import BAML client (generated from schema)
            from baml_client import b
            
            start_time = time.time()
            result = b.ExtractPersonInfo(text)
            execution_time = time.time() - start_time
            
            # Convert BAML result to dictionary
            people_list = []
            for person in result:
                people_list.append({
                    "name": person.name,
                    "age": person.age,
                    "occupation": person.occupation,
                    "location": person.location
                })
            
            extraction_result = {
                "people": people_list,
                "count": len(people_list),
                "extraction_time_ms": round(execution_time * 1000, 2)
            }
            
            baml_logger.log_baml_result(
                "ExtractPersonInfo", 
                True, 
                execution_time,
                people_found=len(people_list),
                person_names=[p["name"] for p in people_list]
            )
            
            return extraction_result
            
        except ImportError:
            error_msg = (
                "BAML client not available. Please run 'baml-cli generate' "
                "to generate the client from the BAML schema files."
            )
            baml_logger.error("BAML client import failed", error=error_msg)
            
            return {
                "error": "BAML client not available",
                "message": error_msg,
                "setup_required": True
            }
            
        except Exception as e:
            error_context = create_error_context(
                e, 
                {"text_length": len(text)}, 
                {"function": "ExtractPersonInfo", "text_preview": text_preview}
            )
            
            baml_logger.log_error_with_context(e, error_context)
            
            return {
                "error": f"Person extraction failed: {str(e)}",
                "error_type": type(e).__name__
            }

# Create tool instances
text_analysis_tool = BAMLTextAnalysisTool()
person_extraction_tool = BAMLPersonExtractionTool()

# Tool registry for easy access
BAML_TOOLS = {
    "text_analysis": text_analysis_tool,
    "extract_people": person_extraction_tool
}

def get_baml_tools() -> List[BaseTool]:
    """Get list of all BAML tools"""
    return list(BAML_TOOLS.values())

def should_use_person_extraction(text: str) -> bool:
    """
    Determine if person extraction tool should be used based on text content.
    """
    person_indicators = [
        "person", "people", "name", "age", "work", "job", "occupation",
        "years old", "employee", "manager", "engineer", "doctor", "teacher",
        "lives in", "from", "works at", "studied at", "mr.", "mrs.", "ms.", "dr."
    ]
    
    text_lower = text.lower()
    has_person_indicators = any(indicator in text_lower for indicator in person_indicators)
    
    # Check for common name patterns (basic heuristic)
    import re
    has_potential_names = bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text))
    
    should_extract = has_person_indicators or has_potential_names
    
    baml_logger.debug(
        "Person extraction decision",
        should_extract=should_extract,
        has_person_indicators=has_person_indicators,
        has_potential_names=has_potential_names,
        text_length=len(text)
    )
    
    return should_extract