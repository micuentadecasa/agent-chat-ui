import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Get the directory where this config.py file is located
_current_dir = Path(__file__).parent

# Try to load .env from the langgraph directory first
_env_file = _current_dir / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file)
    print(f"✓ Loaded environment from: {_env_file}")
else:
    # Fallback to root .env if langgraph/.env doesn't exist
    _root_env = _current_dir.parent / ".env"
    if _root_env.exists():
        load_dotenv(dotenv_path=_root_env)
        print(f"✓ Loaded environment from root: {_root_env}")
    else:
        load_dotenv()  # Load from default locations
        print("⚠ No .env file found, using environment variables and defaults")

class Config:
    """Configuration settings for the LangGraph application"""
    
    # LM Studio Settings
    LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_API_KEY: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # LM Studio doesn't require a real key
    LM_STUDIO_MODEL: str = os.getenv("LM_STUDIO_MODEL", "local-model")  # Default model name
    
    # LangGraph Development Settings
    LANGGRAPH_DEV_PORT: int = int(os.getenv("LANGGRAPH_DEV_PORT", "2024"))
    THREAD_TIMEOUT: int = int(os.getenv("THREAD_TIMEOUT", "300"))  # 5 minutes
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    
    # BAML Settings
    BAML_ENV: str = os.getenv("BAML_ENV", "dev")
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/langgraph.log")
    LOG_JSON_FORMAT: bool = os.getenv("LOG_JSON_FORMAT", "false").lower() == "true"
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration for ChatOpenAI"""
        return {
            "base_url": cls.LM_STUDIO_BASE_URL,
            "api_key": cls.LM_STUDIO_API_KEY,
            "model": cls.LM_STUDIO_MODEL,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
    
    @classmethod
    def get_logging_config(cls) -> dict:
        """Get logging configuration for structured logging"""
        return {
            "log_level": cls.LOG_LEVEL,
            "log_to_file": cls.LOG_TO_FILE,
            "log_file": cls.LOG_FILE,
            "json_format": cls.LOG_JSON_FORMAT,
        }

config = Config()