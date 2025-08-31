// BAML functions for the BAML Tool Node
// This file contains text analysis and information extraction functions

// Data Models for Text Analysis
class TextAnalysis {
  sentiment Sentiment
  key_points KeyPoint[]
  summary string
  word_count int
}

class PersonInfo {
  name string
  age int?
  occupation string?
  location string?
}

// BAML Functions
function AnalyzeText(text: string) -> TextAnalysis {
  client LocalLLM
  prompt #"
    Analyze the following text comprehensively:
    
    Text: {{ text }}
    
    Please provide:
    1. Sentiment analysis with confidence score (0.0 to 1.0)
    2. 3-5 key points from the text with their importance level
    3. A concise summary (2-3 sentences)
    4. Word count
    
    Be thorough but concise in your analysis.
    
    {{ ctx.output_format }}
  "#
}

function ExtractPersonInfo(text: string) -> PersonInfo[] {
  client LocalLLM
  prompt #"
    Extract information about people mentioned in the following text:
    
    {{ text }}
    
    For each person mentioned, extract:
    - Name (required)
    - Age (if mentioned)
    - Occupation (if mentioned) 
    - Location (if mentioned)
    
    {{ ctx.output_format }}
  "#
}

// Test Cases for Text Analysis
test PositiveTextAnalysis {
    functions [AnalyzeText]
    args {
        text "I absolutely love this new project! The team has been incredible and the results exceeded all our expectations. Everyone is excited about the future possibilities."
    }
    @@check(
        sentiment_positive,
        {{ this.sentiment.label == "positive" }}
    )
    @@check(
        confidence_range,
        {{ this.sentiment.confidence >= 0.0 and this.sentiment.confidence <= 1.0 }}
    )
    @@check(
        has_key_points,
        {{ this.key_points|length >= 1 and this.key_points|length <= 5 }}
    )
    @@check(
        has_summary,
        {{ this.summary|length > 0 }}
    )
    @@assert(
        {{ _.checks.sentiment_positive and _.checks.confidence_range and _.checks.has_key_points and _.checks.has_summary }}
    )
    @@assert(
        {{ _.latency_ms < 30000 }}
    )
}

test NegativeTextAnalysis {
    functions [AnalyzeText]
    args {
        text "This is terrible. Nothing is working as expected, the deadlines are impossible, and the whole situation is frustrating everyone involved."
    }
    @@check(
        sentiment_negative,
        {{ this.sentiment.label == "negative" }}
    )
    @@check(
        confidence_valid,
        {{ this.sentiment.confidence >= 0.0 and this.sentiment.confidence <= 1.0 }}
    )
    @@assert(
        {{ _.checks.sentiment_negative and _.checks.confidence_valid }}
    )
}

test NeutralTextAnalysis {
    functions [AnalyzeText]
    args {
        text "The meeting is scheduled for 2 PM tomorrow in conference room A. Please bring your laptops and the quarterly reports. We will review the budget and discuss next steps."
    }
    @@check(
        sentiment_neutral,
        {{ this.sentiment.label == "neutral" }}
    )
    @@assert(
        {{ _.checks.sentiment_neutral }}
    )
}

test EmptyTextHandling {
    functions [AnalyzeText]
    args {
        text ""
    }
    @@check(
        handles_empty,
        {{ this.word_count == 0 }}
    )
    @@assert(
        {{ _.checks.handles_empty }}
    )
}

// Test Cases for Person Information Extraction
test SinglePersonExtraction {
    functions [ExtractPersonInfo]
    args {
        text "John Smith is a 35-year-old software engineer from San Francisco who works at Google."
    }
    @@check(
        found_person,
        {{ this|length == 1 }}
    )
    @@check(
        correct_name,
        {{ this[0].name == "John Smith" }}
    )
    @@check(
        correct_age,
        {{ this[0].age == 35 }}
    )
    @@check(
        has_occupation,
        {{ this[0].occupation != null }}
    )
    @@assert(
        {{ _.checks.found_person and _.checks.correct_name and _.checks.correct_age and _.checks.has_occupation }}
    )
}

test MultiplePersonExtraction {
    functions [ExtractPersonInfo]
    args {
        text "The team includes Sarah Johnson, a 28-year-old data scientist, and Michael Chen, who is 42 and works as a product manager in Seattle."
    }
    @@check(
        found_multiple,
        {{ this|length == 2 }}
    )
    @@check(
        has_names,
        {{ this|map(attribute='name')|list|length == 2 }}
    )
    @@assert(
        {{ _.checks.found_multiple and _.checks.has_names }}
    )
}

test NoPersonMentioned {
    functions [ExtractPersonInfo]
    args {
        text "The weather is nice today. It's sunny with a temperature of 75 degrees. Perfect for a walk in the park."
    }
    @@check(
        no_persons,
        {{ this|length == 0 }}
    )
    @@assert(
        {{ _.checks.no_persons }}
    )
}

test PartialPersonInfo {
    functions [ExtractPersonInfo]
    args {
        text "Alice mentioned that she enjoys reading books, but didn't provide her age or occupation."
    }
    @@check(
        found_partial,
        {{ this|length == 1 }}
    )
    @@check(
        has_name_only,
        {{ this[0].name != null and this[0].age == null }}
    )
    @@assert(
        {{ _.checks.found_partial and _.checks.has_name_only }}
    )
}