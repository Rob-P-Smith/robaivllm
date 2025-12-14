"""
Thinking Tag Preservation Proxy for MiniMax M2.

Transparent proxy that sits between clients and vLLM:
- INBOUND: Injects stored thinking into assistant messages
- OUTBOUND: Copies thinking content to SQLite (passes through unchanged)

This preserves the model's reasoning chain across multi-turn conversations
while allowing clients (like Open WebUI) to handle display stripping.
"""

import os
import re
import json
import httpx
import logging
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from thinking_store import ThinkingStore, FINGERPRINT_LENGTH
from web_search_handler import (
    transform_web_search_tools,
    get_web_search_status,
    WebSearchToolCallAccumulator,
    execute_web_search,
    format_tool_result_for_llm
)

# =============================================================================
# Configuration
# =============================================================================

VLLM_BACKEND = os.getenv("VLLM_BACKEND_URL", "http://localhost:8078")
PROXY_PORT = int(os.getenv("THINKING_PROXY_PORT", "8077"))
DB_PATH = os.getenv("THINKING_DB_PATH", None)
RETENTION_DAYS = int(os.getenv("THINKING_RETENTION_DAYS", "90"))
LOG_LEVEL = os.getenv("THINKING_LOG_LEVEL", "INFO").upper()

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("thinking_proxy")

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Thinking Injection Proxy",
    description="Transparent proxy for MiniMax M2 thinking tag preservation",
    version="1.0.0"
)

# Initialize store
store = ThinkingStore(DB_PATH)

# Regex to extract thinking content
THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)


# =============================================================================
# Core Functions
# =============================================================================

def extract_thinking(text: str) -> tuple[str, str]:
    """
    Extract <think>...</think> from text.

    Args:
        text: Full response text that may contain thinking tags

    Returns:
        Tuple of (thinking_content, cleaned_text)
        If no thinking tags found, returns ("", original_text)
    """
    match = THINK_PATTERN.search(text)
    if match:
        thinking = match.group(1).strip()
        cleaned = THINK_PATTERN.sub('', text).strip()
        return thinking, cleaned
    return "", text


def inject_thinking_into_messages(chat_id: str, messages: list[dict]) -> list[dict]:
    """
    Prepend stored thinking to assistant messages using position-based matching
    with fingerprint verification.

    Open WebUI messages don't include 'id' fields - only 'role' and 'content'.
    We match by position (1st stored → 1st assistant msg) and verify the content
    fingerprint matches to detect conversation branching.

    Args:
        chat_id: The chat identifier
        messages: List of message dicts from the request

    Returns:
        Modified messages list with thinking injected
    """
    # ==========================================================================
    # DEBUG LOG
    # ==========================================================================
    logger.debug(f"[DEBUG-INJECT] Starting injection for chat {chat_id[:8] if chat_id else 'none'}...")
    logger.debug(f"[DEBUG-INJECT] Incoming messages: {len(messages)} total")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    logger.debug(f"[DEBUG-INJECT] Assistant messages in input: {assistant_count}")

    # Get thinking entries with fingerprints, ordered by creation time
    thinking_entries = store.get_ordered_with_fingerprints(chat_id)
    if not thinking_entries:
        logger.debug(f"[DEBUG-INJECT] No stored thinking entries found for this chat")
        return messages

    logger.debug(f"[DEBUG-INJECT] Found {len(thinking_entries)} stored thinking entries")

    result = []
    injected_count = 0
    assistant_index = 0  # Track which assistant message we're on
    first_mismatch_index = None  # Track where conversation branched

    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            if assistant_index < len(thinking_entries):
                stored = thinking_entries[assistant_index]
                # Extract fingerprint from current message content
                # Strip whitespace to match stored fingerprints (extract_thinking uses .strip())
                clean_content = content.strip() if content else ""
                current_fingerprint = clean_content[:FINGERPRINT_LENGTH]

                # Verify fingerprint matches stored fingerprint
                if current_fingerprint == stored["fingerprint"]:
                    # Match! Inject thinking
                    new_msg = msg.copy()
                    # Use <analysis> tags for injection so model recognizes its previous reasoning
                    new_msg["content"] = f"<analysis>\n{stored['thinking']}\n</analysis>\n\n{content}"
                    result.append(new_msg)
                    injected_count += 1
                    logger.debug(f"Injected thinking for assistant msg #{assistant_index}")
                else:
                    # Mismatch! Conversation branched here
                    result.append(msg)
                    if first_mismatch_index is None:
                        first_mismatch_index = assistant_index
                        logger.debug(f"[BRANCH] Detected at position {assistant_index}, skipping injection")
            else:
                # More assistant messages than stored thinking - just pass through
                result.append(msg)

            assistant_index += 1
        else:
            result.append(msg)

    # Prune orphaned entries (from branch point onwards)
    if first_mismatch_index is not None:
        pruned = store.prune_from_position(chat_id, first_mismatch_index)
        if pruned > 0:
            logger.debug(f"[PRUNE] Removed {pruned} orphaned entries for chat {chat_id[:8]}...")

    # ==========================================================================
    # DEBUG LOG
    # ==========================================================================
    logger.debug(f"[DEBUG-INJECT] Injection complete: {injected_count}/{assistant_index} assistant messages got thinking")
    if first_mismatch_index is not None:
        logger.debug(f"[DEBUG-INJECT] Branch detected at position {first_mismatch_index}")

    if injected_count > 0:
        logger.debug(f"[INJECT] Injected thinking into {injected_count}/{assistant_index} assistant messages for chat {chat_id[:8]}...")

    return result


# =============================================================================
# Chat Completions Endpoint
# =============================================================================

async def passthrough_to_vllm(body: dict, is_streaming: bool):
    """
    Pure passthrough to vLLM - no thinking injection or storage.
    Used for auxiliary requests (title/summary/tags).
    """
    if is_streaming:
        async def stream_generator():
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_BACKEND}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{VLLM_BACKEND}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"}
            )
            return response.json()


def is_auxiliary_request(messages: list[dict]) -> bool:
    """
    Detect auxiliary requests (title, summary, tags) that bypass thinking logic.

    Open WebUI sends these with the same message_id as the original chat.
    """
    if not messages:
        return False
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        return False
    content = last_msg.get("content", "")

    # Handle multimodal content (list of content items from Cline/other clients)
    if isinstance(content, list):
        # Extract text from first text item
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                content = item.get("text", "")
                break
        else:
            content = ""  # No text content found

    return isinstance(content, str) and content.strip().startswith("### Task:")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy chat completions with thinking injection/extraction.

    - INBOUND: Injects stored thinking into historical assistant messages
    - OUTBOUND: Copies thinking to DB while passing response unchanged
    - BYPASS: Auxiliary requests (title/summary/tags) pass through untouched
    - AUTONOMOUS: Accumulates thinking across multiple turns into single entry
    """
    body = await request.json()

    # ==========================================================================
    # DEBUG LOG - FULL INCOMING API CALL
    # ==========================================================================
    logger.debug("=" * 80)
    logger.debug("[DEBUG] INCOMING MESSAGE (raw API call to thinking proxy):")
    logger.debug(json.dumps(body, indent=2, default=str))
    logger.debug("=" * 80)

    # Extract thinkingmetadata (minimal data from robaiproxy/robaiagents)
    thinkingmetadata = body.pop("thinkingmetadata", {})
    chat_id = thinkingmetadata.get("chat_id")
    message_id = thinkingmetadata.get("message_id")
    autonomous = thinkingmetadata.get("autonomous", False)  # Autonomous tool-calling mode
    is_streaming = body.get("stream", False)

    # BYPASS: Auxiliary requests go straight through - no injection, no storage
    if is_auxiliary_request(body.get("messages", [])):
        logger.debug(f"Auxiliary request - bypassing thinking logic")
        return await passthrough_to_vllm(body, is_streaming)

    # WEB SEARCH TRANSFORM: Replace malformed web_search_20250305 with standard OpenAI web_search
    # This allows the model to make proper tool calls that route through robairagapi
    # When transformed, we know this is a Claude Code request and need to intercept web_search tool calls
    claude_code_mode = False
    if "tools" in body:
        body["tools"], transformed = transform_web_search_tools(body["tools"])
        if transformed:
            claude_code_mode = True
            logger.info("Transformed web_search_20250305 to standard OpenAI web_search tool (Claude Code mode)")

    # INBOUND: Inject thinking into historical assistant messages
    # Now enabled for autonomous mode too since robaiproxy passes full Open WebUI history
    # Fingerprint matching handles any mismatches gracefully
    if chat_id and "messages" in body:
        body["messages"] = inject_thinking_into_messages(chat_id, body["messages"])

    # ==========================================================================
    # DEBUG LOG - FULL OUTGOING API CALL
    # ==========================================================================
    logger.debug("=" * 80)
    logger.debug("[DEBUG] OUTGOING MESSAGE (API call being sent to vLLM):")
    logger.debug(json.dumps(body, indent=2, default=str))
    logger.debug("=" * 80)

    # Forward to vLLM
    if is_streaming:
        return await handle_streaming(body, chat_id, message_id, autonomous, claude_code_mode)
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            return await handle_non_streaming(client, body, chat_id, message_id, autonomous)


async def handle_streaming(body: dict, chat_id: str, message_id: str, autonomous: bool = False, claude_code_mode: bool = False):
    """
    Handle streaming response - pass through unchanged, copy thinking to DB.

    The response is yielded to the client immediately as chunks arrive.
    We accumulate a copy of the content for thinking extraction after stream completes.

    Uses a line buffer to properly handle SSE lines that span chunk boundaries.
    Pattern matches robaiproxy/requestProxy.py passthrough_stream and
    robaimultiturn/autonomous/tool_loop.py stream_with_tool_extraction.

    For autonomous mode: Appends thinking to existing entry for this message_id,
    allowing multiple autonomous turns to accumulate thinking in one entry.

    For claude_code_mode: Intercepts web_search tool calls, executes via robairagapi,
    then continues the conversation with the search results.

    NOTE: Client is created inside the generator to keep it alive for the
    entire streaming duration. Creating it outside causes "client closed" errors.
    """
    accumulated_content = ""
    line_buffer = ""  # Buffer for incomplete SSE lines spanning chunk boundaries

    async def stream_generator():
        nonlocal accumulated_content, line_buffer

        # Tool call tracking for Claude Code mode
        web_search_accumulator = WebSearchToolCallAccumulator() if claude_code_mode else None

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{VLLM_BACKEND}/v1/chat/completions",
                    json=body,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    async for chunk in response.aiter_bytes():
                        # PARSE for thinking extraction and tool call detection
                        try:
                            chunk_str = chunk.decode('utf-8', errors='ignore')
                            line_buffer += chunk_str

                            # Process complete lines only (split on \n, keep remainder in buffer)
                            while '\n' in line_buffer:
                                line, line_buffer = line_buffer.split('\n', 1)
                                line = line.strip()

                                if not line:
                                    continue

                                if line.startswith('data: '):
                                    data_str = line[6:]

                                    if data_str == '[DONE]':
                                        continue  # Stream done marker - skip

                                    try:
                                        data = json.loads(data_str)
                                        choices = data.get('choices', [])
                                        if choices:
                                            delta = choices[0].get('delta', {})
                                            finish_reason = choices[0].get('finish_reason')

                                            # Accumulate content for thinking extraction
                                            content = delta.get('content', '')
                                            if content:
                                                accumulated_content += content

                                            # Claude Code mode: track web_search tool calls
                                            if claude_code_mode and web_search_accumulator:
                                                tool_complete = web_search_accumulator.process_delta(delta, finish_reason)

                                                if tool_complete and web_search_accumulator.tool_name == 'web_search':
                                                    # Web search tool call detected - call web_search_20250305 agent to get results
                                                    args = web_search_accumulator.get_parsed_arguments()
                                                    query = args.get('query', '')

                                                    # Extract original user prompt
                                                    original_prompt = next(
                                                        (m["content"] for m in reversed(body.get("messages", []))
                                                         if m.get("role") == "user"),
                                                        ""
                                                    )

                                                    logger.info(f"Calling web_search_20250305 agent for: {query}")

                                                    # Call web_search_20250305 agent via HTTP API (no cross-project imports)
                                                    try:
                                                        agents_url = os.getenv("AGENTS_API_URL", "http://localhost:8090")
                                                        api_key = os.getenv("REST_API_KEY", "")

                                                        agent_config = {
                                                            "vllm_url": VLLM_BACKEND,
                                                            "v2_base_url": os.getenv("ROBAIRAGAPI_URL", "http://localhost:8081") + "/api/v2",
                                                            "v2_api_key": api_key,
                                                            "model_name": body.get("model", "Qwen3-30B"),
                                                            "original_prompt": original_prompt,
                                                            "search_query": query,
                                                            "tool_call_id": web_search_accumulator.tool_call_id
                                                        }

                                                        logger.info(f"Calling web_search_20250305 agent via HTTP: {agents_url}")

                                                        # Call robaiagents HTTP API to execute agent
                                                        agent_payload = {
                                                            "agent_type": "web_search_20250305",
                                                            "task": query,
                                                            "config": agent_config
                                                        }

                                                        async with httpx.AsyncClient(timeout=120.0) as agent_client:
                                                            async with agent_client.stream(
                                                                "POST",
                                                                f"{agents_url}/api/v1/agents/run",
                                                                json=agent_payload,
                                                                headers={"Authorization": f"Bearer {api_key}"}
                                                            ) as agent_response:
                                                                agent_response.raise_for_status()

                                                                # Agent returns dict with search results + crawled content
                                                                agent_response_text = ""
                                                                async for agent_chunk in agent_response.aiter_bytes():
                                                                    agent_response_text += agent_chunk.decode('utf-8', errors='ignore')

                                                        # Parse agent response (it's a JSON dict)
                                                        agent_result = json.loads(agent_response_text)

                                                        if not agent_result.get("success"):
                                                            logger.error(f"Agent failed: {agent_result.get('error')}")
                                                            # Stream error message and exit
                                                            error_data = {
                                                                "choices": [{
                                                                    "delta": {"content": f"Error: {agent_result.get('error')}"},
                                                                    "finish_reason": "stop"
                                                                }]
                                                            }
                                                            yield f"data: {json.dumps(error_data)}\n\n".encode()
                                                            return

                                                        # Agent returned search results + crawled content
                                                        search_results = agent_result.get("search_results", [])
                                                        crawled_content = agent_result.get("crawled_content", [])

                                                        logger.info(f"Agent returned {len(search_results)} search results and {len(crawled_content)} crawled URLs")

                                                        # Now build NEW request to vLLM with search results in context
                                                        # This keeps vLLM in the context and lets it generate the response
                                                        from web_search_prompt import build_system_prompt

                                                        system_prompt = build_system_prompt(
                                                            query=query,
                                                            search_results=search_results,
                                                            crawled_content=crawled_content
                                                        )

                                                        followup_messages = [
                                                            {"role": "system", "content": system_prompt},
                                                            {"role": "user", "content": original_prompt}
                                                        ]

                                                        logger.info("Sending follow-up request to vLLM with search context injected")

                                                        # Make follow-up request to vLLM (no tools this time, just context)
                                                        followup_body = {
                                                            "model": body.get("model"),
                                                            "messages": followup_messages,
                                                            "stream": True,
                                                            "max_tokens": body.get("max_tokens", 4000),
                                                            "temperature": body.get("temperature", 0.7)
                                                        }

                                                        # Stream the vLLM response back as continuation of original conversation
                                                        async with client.stream(
                                                            "POST",
                                                            f"{VLLM_BACKEND}/v1/chat/completions",
                                                            json=followup_body,
                                                            headers={"Content-Type": "application/json"}
                                                        ) as followup_response:
                                                            async for followup_chunk in followup_response.aiter_bytes():
                                                                # Parse chunk to accumulate content
                                                                try:
                                                                    chunk_str = followup_chunk.decode('utf-8', errors='ignore')
                                                                    for line in chunk_str.split('\n'):
                                                                        if line.startswith('data: '):
                                                                            data_str = line[6:]
                                                                            if data_str != '[DONE]':
                                                                                data = json.loads(data_str)
                                                                                choices = data.get('choices', [])
                                                                                if choices:
                                                                                    delta = choices[0].get('delta', {})
                                                                                    content = delta.get('content', '')
                                                                                    if content:
                                                                                        accumulated_content += content
                                                                except:
                                                                    pass

                                                                # Yield chunk as-is (vLLM OpenAI format)
                                                                yield followup_chunk

                                                        # Done - exit original stream
                                                        return

                                                    except Exception as agent_error:
                                                        logger.error(f"web_search_20250305 agent error: {agent_error}", exc_info=True)
                                                        # Fallback: return error message
                                                        error_data = {
                                                            "choices": [{
                                                                "delta": {"content": f"Error: {str(agent_error)}"},
                                                                "finish_reason": "stop"
                                                            }]
                                                        }
                                                        yield f"data: {json.dumps(error_data)}\n\n".encode()
                                                        return

                                    except json.JSONDecodeError:
                                        pass  # Incomplete or malformed JSON - skip
                        except Exception as e:
                            logger.debug(f"Error processing chunk: {e}")
                            pass  # Don't break passthrough on parsing errors

                        # Yield original chunk (passthrough)
                        yield chunk

        except httpx.HTTPError as e:
            logger.error(f"HTTP error forwarding to vLLM: {e}")
            error_response = {
                "error": {
                    "message": f"Backend error: {str(e)}",
                    "type": "proxy_error"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n".encode()
            return

        # ==========================================================================
        # DEBUG LOG - FULL RESPONSE FROM vLLM
        # ==========================================================================
        logger.debug("=" * 80)
        logger.debug("[DEBUG] RESPONSE MESSAGE (accumulated streaming response from vLLM):")
        logger.debug(f"[DEBUG] chat_id: {chat_id}, message_id: {message_id}, autonomous: {autonomous}")
        logger.debug(f"[DEBUG] accumulated_content ({len(accumulated_content)} chars):")
        logger.debug(accumulated_content)
        logger.debug("=" * 80)

        # After stream complete - extract and store thinking content with fingerprint
        if chat_id and message_id and accumulated_content:
            thinking, cleaned_content = extract_thinking(accumulated_content)
            if thinking:
                if autonomous:
                    # Autonomous mode: append to existing entry (accumulate across turns)
                    appended = store.store_or_append(chat_id, message_id, thinking, cleaned_content)
                    action = "APPEND" if appended else "STORE"
                    logger.debug(f"[{action}] Autonomous thinking for msg {message_id[:8]}... ({len(thinking)} chars)")
                else:
                    # Normal mode: replace entry
                    store.store(chat_id, message_id, thinking, cleaned_content)
                    logger.debug(f"[STORE] Saved thinking for msg {message_id[:8]}... ({len(thinking)} chars)")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


async def handle_non_streaming(client: httpx.AsyncClient, body: dict, chat_id: str, message_id: str, autonomous: bool = False):
    """
    Handle non-streaming response - return unchanged, copy thinking to DB.

    For autonomous mode: Appends thinking to existing entry for this message_id.
    """
    try:
        response = await client.post(
            f"{VLLM_BACKEND}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
    except httpx.HTTPError as e:
        logger.error(f"HTTP error forwarding to vLLM: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Backend error: {str(e)}", "type": "proxy_error"}}
        )

    # ==========================================================================
    # DEBUG LOG - FULL RESPONSE FROM vLLM
    # ==========================================================================
    logger.debug("=" * 80)
    logger.debug("[DEBUG] RESPONSE MESSAGE (non-streaming response from vLLM):")
    logger.debug(f"[DEBUG] chat_id: {chat_id}, message_id: {message_id}, autonomous: {autonomous}")
    logger.debug(json.dumps(result, indent=2, default=str))
    logger.debug("=" * 80)

    # Copy thinking to DB with fingerprint (response returned unchanged to client)
    if chat_id and message_id:
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            thinking, cleaned_content = extract_thinking(content)
            if thinking:
                if autonomous:
                    # Autonomous mode: append to existing entry (accumulate across turns)
                    appended = store.store_or_append(chat_id, message_id, thinking, cleaned_content)
                    action = "APPEND" if appended else "STORE"
                    logger.debug(f"[{action}] Autonomous thinking for msg {message_id[:8]}... ({len(thinking)} chars)")
                else:
                    # Normal mode: replace entry
                    store.store(chat_id, message_id, thinking, cleaned_content)
                    logger.debug(f"[STORE] Saved thinking for msg {message_id[:8]}... ({len(thinking)} chars)")

    # Return full response unchanged - client gets <think> tags
    return result


# =============================================================================
# Health & Stats Endpoints (MUST be before catch-all passthrough)
# =============================================================================

@app.get("/thinking/health")
async def health():
    """Health check endpoint for the thinking proxy."""
    return {
        "status": "healthy",
        "service": "thinking-proxy",
        "backend": VLLM_BACKEND
    }


@app.get("/thinking/stats")
async def stats():
    """Get thinking storage statistics."""
    return store.get_stats()


@app.post("/thinking/cleanup")
async def cleanup():
    """Trigger cleanup of old thinking records."""
    deleted = store.cleanup_old(days=RETENTION_DAYS)
    return {"deleted": deleted, "retention_days": RETENTION_DAYS}


# =============================================================================
# Passthrough Endpoint (all other routes) - MUST be last
# =============================================================================

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_passthrough(request: Request, path: str):
    """Pass through all other requests unchanged to vLLM."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{VLLM_BACKEND}/{path}"

        try:
            if request.method == "GET":
                response = await client.get(url, params=request.query_params)
            else:
                body = await request.body()
                response = await client.request(
                    request.method,
                    url,
                    content=body,
                    headers={"Content-Type": request.headers.get("content-type", "application/json")}
                )

            content_type = response.headers.get("content-type", "")
            if content_type.startswith("application/json"):
                return response.json()
            else:
                return response.content

        except httpx.HTTPError as e:
            logger.error(f"HTTP error on passthrough to {path}: {e}")
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Backend error: {str(e)}", "type": "proxy_error"}}
            )


# =============================================================================
# Startup/Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup():
    """Startup tasks."""
    web_search_status = get_web_search_status()

    logger.info("=" * 60)
    logger.info("THINKING PROXY STARTING")
    logger.info("=" * 60)
    logger.info(f"Backend URL: {VLLM_BACKEND}")
    logger.info(f"Proxy Port:  {PROXY_PORT}")
    logger.info(f"DB Path:     {store.db_path}")
    logger.info(f"Retention:   {RETENTION_DAYS} days")
    logger.info(f"Log Level:   {LOG_LEVEL}")
    logger.info(f"Web Search:  {web_search_status.get('mode', 'unknown')} - {web_search_status.get('description', '')}")
    logger.info("=" * 60)

    # Run initial cleanup
    deleted = store.cleanup_old(days=RETENTION_DAYS)
    if deleted > 0:
        logger.info(f"Startup cleanup: removed {deleted} old records")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
