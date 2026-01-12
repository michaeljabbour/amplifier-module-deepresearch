"""Test fixtures for deep research module."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Create a mock Anthropic client."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def openai_research_response() -> MagicMock:
    """Create a mock OpenAI deep research response."""
    response = MagicMock()
    response.id = "resp_test123"
    response.status = "completed"

    # Mock usage
    response.usage = MagicMock()
    response.usage.input_tokens = 1000
    response.usage.output_tokens = 5000

    # Mock output items
    # 1. Reasoning step
    reasoning_item = MagicMock()
    reasoning_item.type = "reasoning"
    reasoning_item.summary = [MagicMock(text="Analyzing the research question...")]

    # 2. Web search step
    web_search_item = MagicMock()
    web_search_item.type = "web_search_call"
    web_search_item.action = {"type": "search", "query": "economic impact of AI"}
    web_search_item.status = "completed"

    # 3. Final message with citations
    message_item = MagicMock()
    message_item.type = "message"

    text_content = MagicMock()
    text_content.type = "output_text"
    text_content.text = "# Research Report\n\nAI has significant economic impact..."

    annotation = MagicMock()
    annotation.title = "AI Economic Impact Study"
    annotation.url = "https://example.com/ai-study"
    annotation.start_index = 50
    annotation.end_index = 100
    text_content.annotations = [annotation]

    message_item.content = [text_content]

    response.output = [reasoning_item, web_search_item, message_item]

    return response


@pytest.fixture
def anthropic_research_response() -> MagicMock:
    """Create a mock Anthropic research response."""
    response = MagicMock()
    response.stop_reason = "end_turn"

    # Mock usage
    response.usage = MagicMock()
    response.usage.input_tokens = 800
    response.usage.output_tokens = 3000

    # Mock content blocks
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = "Let me analyze this research question step by step..."
    thinking_block.signature = "sig123"

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "# Research Findings\n\nBased on my research, [Source](https://example.com)..."

    response.content = [thinking_block, text_block]

    return response


@pytest.fixture
def sample_research_query() -> str:
    """Sample research query for testing."""
    return "Research the economic impact of artificial intelligence on global healthcare systems."


@pytest.fixture
def sample_instructions() -> str:
    """Sample research instructions."""
    return """
    Focus on data-rich insights including specific figures and statistics.
    Prioritize reliable sources: peer-reviewed research and health organizations.
    Include inline citations and source metadata.
    """


def create_openai_response_with_file_search() -> MagicMock:
    """Create mock response with file search results."""
    response = MagicMock()
    response.id = "resp_file123"
    response.status = "completed"
    response.usage = MagicMock(input_tokens=1500, output_tokens=6000)

    # File search item
    file_search_item = MagicMock()
    file_search_item.type = "file_search_call"
    file_search_item.query = "internal research documents"
    file_search_item.status = "completed"

    # Final message
    message_item = MagicMock()
    message_item.type = "message"
    text_content = MagicMock()
    text_content.type = "output_text"
    text_content.text = "Based on internal documents and web research..."
    text_content.annotations = []
    message_item.content = [text_content]

    response.output = [file_search_item, message_item]
    return response


def create_openai_response_with_code_execution() -> MagicMock:
    """Create mock response with code interpreter results."""
    response = MagicMock()
    response.id = "resp_code123"
    response.status = "completed"
    response.usage = MagicMock(input_tokens=2000, output_tokens=8000)

    # Code execution item
    code_item = MagicMock()
    code_item.type = "code_interpreter_call"
    code_item.input = "import pandas as pd\ndf = pd.read_csv('data.csv')"
    code_item.output = "DataFrame with 1000 rows loaded"

    # Final message
    message_item = MagicMock()
    message_item.type = "message"
    text_content = MagicMock()
    text_content.type = "output_text"
    text_content.text = "Analysis complete. The data shows..."
    text_content.annotations = []
    message_item.content = [text_content]

    response.output = [code_item, message_item]
    return response
