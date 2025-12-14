"""
Web Search Tool Transformer for Claude Code web_search_20250305.

Transforms Claude Code's malformed web_search_20250305 tool definition into
a proper OpenAI-format web_search tool that matches robaiLLMtools.

The transformed tool call is then routed through the normal tool execution
path via robairagapi, which already handles web_search via Serper.
"""

import os
import json
import logging
import httpx
from typing import Optional, Dict, Any

logger = logging.getLogger("thinking_proxy.web_search")

# =============================================================================
# Configuration
# =============================================================================

ROBAIRAGAPI_URL = os.getenv("ROBAIRAGAPI_URL", "http://localhost:8081")
ROBAIRAGAPI_KEY = os.getenv("REST_API_KEY", "")


# =============================================================================
# Standard web_search Tool Definition (from robaiLLMtools/tooldiscovery.py)
# =============================================================================

STANDARD_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web using Serper API (Google Search). Returns titles, URLs, and brief snippets. USE THIS TOOL TO DISCOVER relevant sources, then use crawl_url to fetch full page content from the best URLs. This is a discovery tool - snippets are brief, so always follow up with crawl_url for detailed information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string (e.g., 'python async patterns', 'react hooks best practices')"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return, 1-20 (default: 10)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 20
                },
                "max_chars_per_result": {
                    "type": "integer",
                    "description": "Maximum characters per result snippet (200-15000, default: 500). Keep low since this is for discovery - use crawl_url for full content.",
                    "default": 500,
                    "minimum": 200,
                    "maximum": 15000
                }
            },
            "required": ["query"]
        }
    }
}


# =============================================================================
# Tool Detection Functions
# =============================================================================

def is_claude_code_web_search(tool: dict) -> bool:
    """
    Detect if this is a Claude Code web_search_20250305 tool.

    Distinguishes from robaiagents' normal web_search tool by checking
    for the specific 'web_search_20250305' type in parameters.

    Handles two formats:
    1. OpenAI function format: {"type": "function", "function": {"name": "web_search", "parameters": {...}}}
    2. Anthropic native format: {"name": "web_search", "parameters": {"type": "web_search_20250305", ...}}

    Args:
        tool: Tool definition dict from tools array

    Returns:
        True if this is web_search_20250305
    """
    # Format 1: OpenAI function wrapper
    if tool.get("type") == "function":
        function = tool.get("function", {})
        if function.get("name") != "web_search":
            return False
        parameters = function.get("parameters", {})
        return parameters.get("type") == "web_search_20250305"

    # Format 2: Anthropic native format (no function wrapper)
    # Tool has direct "name" and "parameters" fields
    if tool.get("name") == "web_search":
        parameters = tool.get("parameters", {})
        return parameters.get("type") == "web_search_20250305"

    return False


def transform_web_search_tools(tools: list) -> tuple[list, bool]:
    """
    Transform web_search_20250305 tool to standard OpenAI web_search format.

    Replaces the malformed Claude Code web_search_20250305 tool definition
    with a proper OpenAI-format tool that the model can actually call.

    Args:
        tools: Tools array from request

    Returns:
        (transformed_tools, was_transformed)
    """
    if not tools:
        return tools, False

    transformed = False
    new_tools = []

    for tool in tools:
        if is_claude_code_web_search(tool):
            # Replace with standard web_search tool
            new_tools.append(STANDARD_WEB_SEARCH_TOOL)
            transformed = True
            logger.info("Transformed web_search_20250305 to standard OpenAI web_search tool")
        else:
            new_tools.append(tool)

    return new_tools, transformed


# =============================================================================
# Tool Call Accumulator for Streaming
# =============================================================================

class WebSearchToolCallAccumulator:
    """
    Accumulates web_search tool call data from streaming chunks.

    Tracks when a web_search tool call starts, accumulates arguments,
    and signals when it's complete.
    """

    def __init__(self):
        self.tool_call_id: Optional[str] = None
        self.tool_name: Optional[str] = None
        self.arguments: str = ""
        self.is_active: bool = False
        self.is_complete: bool = False

    def reset(self):
        """Reset state for next tool call."""
        self.tool_call_id = None
        self.tool_name = None
        self.arguments = ""
        self.is_active = False
        self.is_complete = False

    def process_delta(self, delta: dict, finish_reason: Optional[str] = None) -> bool:
        """
        Process a streaming delta to track web_search tool calls.

        Args:
            delta: The delta object from the streaming response
            finish_reason: The finish_reason if present

        Returns:
            True if a web_search tool call just completed
        """
        tool_calls = delta.get('tool_calls', [])

        for tool_call in tool_calls:
            function_info = tool_call.get('function', {})

            # Detect tool call start (name is only sent in first chunk)
            tool_name = function_info.get('name', '')
            if tool_name:
                self.tool_name = tool_name
                self.tool_call_id = tool_call.get('id') or self.tool_call_id
                if tool_name == 'web_search':
                    self.is_active = True
                    logger.debug(f"Detected web_search tool call: {self.tool_call_id}")

            # Accumulate arguments (they come in chunks)
            if self.is_active and 'arguments' in function_info:
                self.arguments += function_info['arguments']

        # Check if tool call is complete
        if finish_reason == 'tool_calls' and self.is_active:
            self.is_complete = True
            logger.debug(f"web_search tool call complete: {self.arguments}")
            return True

        return False

    def get_parsed_arguments(self) -> dict:
        """
        Parse accumulated arguments JSON.

        Returns:
            Parsed arguments dict, or empty dict on error
        """
        try:
            return json.loads(self.arguments) if self.arguments else {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse web_search arguments: {e}")
            logger.debug(f"Raw arguments: {self.arguments}")
            return {}


# =============================================================================
# Execute Web Search via robairagapi
# =============================================================================

async def execute_web_search(
    query: str,
    num_results: int = 10,
    max_chars_per_result: int = 500,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Execute web search via robairagapi /api/v2/web_search endpoint.

    Args:
        query: Search query string
        num_results: Number of results to return
        max_chars_per_result: Max characters per result snippet
        timeout: Request timeout in seconds

    Returns:
        Search results dict from robairagapi
    """
    if not ROBAIRAGAPI_KEY:
        logger.error("REST_API_KEY not configured for robairagapi")
        return {
            "success": False,
            "error": "CONFIGURATION_ERROR",
            "message": "REST_API_KEY not configured"
        }

    url = f"{ROBAIRAGAPI_URL}/api/v2/web_search"
    headers = {
        "Authorization": f"Bearer {ROBAIRAGAPI_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "num_results": num_results,
        "max_chars_per_result": max_chars_per_result
    }

    try:
        logger.info(f"Executing web search via robairagapi: {query}")
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        logger.info(f"Web search returned {result.get('total_results', 0)} results")
        return result

    except httpx.TimeoutException:
        logger.error(f"Web search timed out for: {query}")
        return {
            "success": False,
            "error": "TIMEOUT",
            "message": "Web search timed out"
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"Web search HTTP error: {e.response.status_code}")
        return {
            "success": False,
            "error": "HTTP_ERROR",
            "message": f"robairagapi error: {e.response.status_code}"
        }

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "success": False,
            "error": "EXECUTION_FAILED",
            "message": f"Web search failed: {str(e)}"
        }


def format_tool_result_for_llm(tool_call_id: str, search_result: Dict[str, Any]) -> str:
    """
    Format search results as a string for the LLM tool result message.

    Args:
        tool_call_id: The tool call ID
        search_result: Results from execute_web_search

    Returns:
        Formatted string for tool result content
    """
    if not search_result.get("success"):
        return f"Web search failed: {search_result.get('message', 'Unknown error')}"

    results = search_result.get("results", [])
    if not results:
        return f"Web search for '{search_result.get('query', '')}' returned no results."

    # Format results for the LLM
    lines = [f"Web search results for: \"{search_result.get('query', '')}\"\n"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {link}")
        if snippet:
            lines.append(f"   {snippet[:200]}...")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Status Reporting
# =============================================================================

def get_web_search_status() -> dict:
    """
    Get status information about web search capability.

    Returns:
        Status dict for health/stats endpoints
    """
    return {
        "enabled": True,
        "mode": "tool_transform",
        "robairagapi_url": ROBAIRAGAPI_URL,
        "api_key_configured": bool(ROBAIRAGAPI_KEY),
        "description": "Transforms web_search_20250305 to standard web_search for robairagapi execution"
    }
