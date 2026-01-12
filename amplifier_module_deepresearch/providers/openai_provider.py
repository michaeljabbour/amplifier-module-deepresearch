"""OpenAI Deep Research provider implementation.

Supports OpenAI's native Deep Research API with:
- o3-deep-research: Full-featured, comprehensive reports
- o4-mini-deep-research: Faster, cost-effective option

Key features:
- Web search (web_search_preview)
- File search over vector stores
- Code interpreter for data analysis
- MCP servers for private data access
- Background mode for long-running tasks
- max_tool_calls for cost/latency control
"""

import asyncio
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from .._constants import (
    OPENAI_DEFAULT_MAX_TOKENS,
    OPENAI_DEFAULT_MODEL,
    OPENAI_DEFAULT_TIMEOUT,
    OPENAI_MODELS,
)
from .base import (
    Citation,
    CodeExecutionStep,
    DeepResearchProviderBase,
    DeepResearchRequest,
    DeepResearchResult,
    FileSearchStep,
    ReasoningStep,
    WebSearchStep,
)

logger = logging.getLogger(__name__)


class OpenAIDeepResearchProvider(DeepResearchProviderBase):
    """OpenAI Deep Research API provider.

    Uses OpenAI's Responses API with specialized deep research models
    that autonomously conduct multi-step research.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
        timeout: float = OPENAI_DEFAULT_TIMEOUT,
        debug: bool = False,
    ):
        """Initialize OpenAI Deep Research provider.

        Args:
            api_key: OpenAI API key
            client: Pre-configured AsyncOpenAI client (for testing)
            timeout: Default request timeout in seconds
            debug: Enable debug logging
        """
        if client is not None:
            self._client = client
            self._api_key = None
        elif api_key is not None:
            self._client = AsyncOpenAI(api_key=api_key)
            self._api_key = api_key
        else:
            self._client = None
            self._api_key = None

        self.timeout = timeout
        self.debug = debug

    @property
    def is_available(self) -> bool:
        """Check if provider has valid credentials."""
        return self._client is not None

    async def list_models(self) -> list[dict[str, Any]]:
        """List available Deep Research models."""
        return [
            {
                "id": "o3-deep-research",
                "display_name": "O3 Deep Research",
                "context_window": 200000,
                "max_output_tokens": 100000,
                "capabilities": ["deep_research", "citations", "web_search", "file_search", "code_interpreter"],
                "description": "Comprehensive research with highest quality synthesis",
            },
            {
                "id": "o3-deep-research-2025-06-26",
                "display_name": "O3 Deep Research (2025-06-26)",
                "context_window": 200000,
                "max_output_tokens": 100000,
                "capabilities": ["deep_research", "citations", "web_search", "file_search", "code_interpreter"],
                "description": "Dated snapshot of O3 Deep Research",
            },
            {
                "id": "o4-mini-deep-research",
                "display_name": "O4 Mini Deep Research",
                "context_window": 200000,
                "max_output_tokens": 50000,
                "capabilities": ["deep_research", "citations", "web_search", "file_search", "code_interpreter", "fast"],
                "description": "Faster, cost-effective research for latency-sensitive tasks",
            },
            {
                "id": "o4-mini-deep-research-2025-06-26",
                "display_name": "O4 Mini Deep Research (2025-06-26)",
                "context_window": 200000,
                "max_output_tokens": 50000,
                "capabilities": ["deep_research", "citations", "web_search", "file_search", "code_interpreter", "fast"],
                "description": "Dated snapshot of O4 Mini Deep Research",
            },
        ]

    async def research(self, request: DeepResearchRequest) -> DeepResearchResult:
        """Execute a deep research request via OpenAI's Responses API.

        Args:
            request: Research request with query and configuration

        Returns:
            DeepResearchResult with report and metadata
        """
        if not self._client:
            raise RuntimeError("OpenAI client not initialized - missing API key")

        model = request.model or OPENAI_DEFAULT_MODEL
        if model not in OPENAI_MODELS:
            logger.warning(f"Model {model} not in known models list: {OPENAI_MODELS}")

        # Build input messages
        input_messages = self._build_input(request)

        # Build tools configuration
        tools = self._build_tools(request)

        # Build request parameters
        params: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "reasoning": {"summary": request.reasoning_summary},
        }

        if tools:
            params["tools"] = tools

        if request.max_output_tokens:
            params["max_output_tokens"] = request.max_output_tokens
        else:
            params["max_output_tokens"] = OPENAI_DEFAULT_MAX_TOKENS

        # Background mode for long-running tasks
        if request.background:
            params["background"] = True

        # Limit tool calls for cost/latency control
        if request.max_tool_calls:
            params["max_tool_calls"] = request.max_tool_calls

        timeout = request.timeout or self.timeout

        logger.info(
            f"[OpenAI Deep Research] Starting research with model={model}, tools={[t.get('type') for t in tools]}"
        )

        start_time = time.time()

        try:
            response = await asyncio.wait_for(
                self._client.responses.create(**params),
                timeout=timeout,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[OpenAI Deep Research] Completed in {elapsed_ms}ms")

            return self._parse_response(response, model)

        except TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[OpenAI Deep Research] Timeout after {elapsed_ms}ms (limit: {timeout}s)")
            raise

    def _build_input(self, request: DeepResearchRequest) -> list[dict[str, Any]]:
        """Build input messages for the Responses API."""
        messages = []

        # Developer instructions (system message equivalent)
        if request.instructions:
            messages.append(
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": request.instructions}],
                }
            )

        # User query
        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": request.query}],
            }
        )

        return messages

    def _build_tools(self, request: DeepResearchRequest) -> list[dict[str, Any]]:
        """Build tools configuration for deep research.

        Deep Research requires at least one data source:
        - web_search_preview (default)
        - file_search with vector stores
        - MCP servers
        """
        tools = []

        # Web search is the default and most common tool
        if request.enable_web_search:
            tools.append({"type": "web_search_preview"})

        # File search requires vector store IDs
        if request.enable_file_search and request.vector_store_ids:
            tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": request.vector_store_ids[:2],  # Max 2 vector stores
                }
            )

        # Code interpreter for data analysis
        if request.enable_code_interpreter:
            tools.append(
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto"},
                }
            )

        # MCP servers for private data
        # Note: Deep Research requires MCP servers with search/fetch interface
        for server in request.mcp_servers:
            tools.append(
                {
                    "type": "mcp",
                    "server_label": server.get("label", "mcp"),
                    "server_url": server.get("url"),
                    "require_approval": "never",  # Required for Deep Research
                }
            )

        return tools

    def _parse_response(self, response: Any, model: str) -> DeepResearchResult:
        """Parse OpenAI Responses API response into DeepResearchResult."""
        # Extract final report text
        report_text = ""
        citations: list[Citation] = []
        reasoning_steps: list[ReasoningStep] = []
        web_searches: list[WebSearchStep] = []
        file_searches: list[FileSearchStep] = []
        code_executions: list[CodeExecutionStep] = []

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "message":
                # Final answer with citations
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        report_text = getattr(content, "text", "")

                        # Extract citations from annotations
                        for ann in getattr(content, "annotations", []):
                            citations.append(
                                Citation(
                                    title=getattr(ann, "title", ""),
                                    url=getattr(ann, "url", ""),
                                    start_index=getattr(ann, "start_index", 0),
                                    end_index=getattr(ann, "end_index", 0),
                                )
                            )

            elif item_type == "reasoning":
                # Reasoning/thinking steps
                for summary in getattr(item, "summary", []):
                    text = summary.get("text", "") if isinstance(summary, dict) else getattr(summary, "text", "")
                    if text:
                        reasoning_steps.append(ReasoningStep(text=text))

            elif item_type == "web_search_call":
                # Web search actions
                action = getattr(item, "action", {})
                if isinstance(action, dict):
                    query = action.get("query", "")
                    action_type = action.get("type", "search")
                else:
                    query = getattr(action, "query", "")
                    action_type = getattr(action, "type", "search")

                web_searches.append(
                    WebSearchStep(
                        query=query,
                        status=getattr(item, "status", "completed"),
                        action_type=action_type,
                    )
                )

            elif item_type == "file_search_call":
                # File search over vector stores
                query = getattr(item, "query", "")
                file_searches.append(
                    FileSearchStep(
                        query=query,
                        status=getattr(item, "status", "completed"),
                    )
                )

            elif item_type == "code_interpreter_call":
                # Code execution
                code_executions.append(
                    CodeExecutionStep(
                        input_code=getattr(item, "input", ""),
                        output=getattr(item, "output", None),
                    )
                )

        # Extract usage
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        return DeepResearchResult(
            report_text=report_text,
            provider="openai",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_id=getattr(response, "id", None),
            status=getattr(response, "status", None),
            citations=citations,
            reasoning_steps=reasoning_steps,
            web_searches=web_searches,
            file_searches=file_searches,
            code_executions=code_executions,
            raw_response=response if self.debug else None,
        )
