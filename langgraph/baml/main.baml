// Shared BAML Configuration
generator lang_python {
  output_type python/pydantic
  output_dir "../"
  version "0.65.0"
}

client LocalLLM {
  provider openai
  options {
    base_url env.LM_STUDIO_BASE_URL
    api_key env.LM_STUDIO_API_KEY
    model env.LM_STUDIO_MODEL
  }
}

// Shared Data Models
class Sentiment {
  label "positive" | "negative" | "neutral"
  confidence float
}

class KeyPoint {
  point string
  importance "high" | "medium" | "low"
}