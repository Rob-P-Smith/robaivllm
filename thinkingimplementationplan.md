# Thinking Tag Preservation System for MiniMax M2

## Overview

Implement a **transparent thinking proxy** that sits between all clients and vLLM. It extracts `<think>` tags from model responses, stores them indexed by chat_id and message position, and re-injects them into subsequent requests so the model maintains its reasoning chain across multi-turn conversations.

## Key Requirements (from MiniMax docs)

> "You must preserve the model's thinking content completely, i.e., `<think>reasoning_content</think>`. This is essential to ensure Interleaved Thinking works effectively."

**Format:** `<think>\nreasoning\n</think>\n\ncontent`

## Architecture

```
                   ┌─────────────────────────────────────────┐
                   │  Clients (Open WebUI, robaiproxy, etc)  │
                   │  → All continue pointing to port 8078   │
                   └─────────────────────┬───────────────────┘
                                         │
                                         ▼
                   ┌─────────────────────────────────────────┐
                   │  robaivllm/ Thinking Proxy (Port 8078)  │
                   │  - Injects stored thinking on INBOUND   │
                   │  - Extracts thinking on OUTBOUND        │
                   │  - SQLite persistence for chat history  │
                   └─────────────────────┬───────────────────┘
                                         │
                                         ▼
                   ┌─────────────────────────────────────────┐
                   │  vLLM (Port 8077 - moved from 8078)     │
                   └─────────────────────────────────────────┘
```

**File Structure:**
```
robaivllm/
├── __init__.py
├── thinking_proxy.py       # FastAPI app, main proxy logic
├── thinking_store.py       # SQLite storage layer
├── data/
│   └── thinking.db         # SQLite database (auto-created)
└── Dockerfile              # Optional containerization
```

## Message Structure (from Open WebUI)

Each message in the messages array contains:
- `id`: Unique message ID (e.g., "9e16c3de-e624-49ab-be6e-7dd0d0e4ff13")
- `parentId`: Parent message ID (links assistant to user message)
- `role`: "user" | "assistant" | "system"
- `content`: Message content

Metadata includes:
- `chat_id`: Unique chat identifier
- `message_id`: Current message ID

**Injection Strategy:** Use message `id` to align thinking with the correct assistant response in the conversation history.

## Implementation Plan

### File 1: `robaivllm/thinking_store.py`

**Purpose:** SQLite storage layer for thinking content

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS thinking_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    message_id TEXT NOT NULL,           -- Links to specific assistant message
    thinking_content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_thinking_chat_id ON thinking_history(chat_id);
CREATE INDEX IF NOT EXISTS idx_thinking_created ON thinking_history(created_at);
```

**Class: `ThinkingStore`**
```python
class ThinkingStore:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "thinking.db"
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None
    def store(self, chat_id: str, message_id: str, thinking: str) -> None
    def get_for_message(self, chat_id: str, message_id: str) -> Optional[str]
    def get_all_for_chat(self, chat_id: str) -> dict[str, str]  # {message_id: thinking}
    def delete_chat(self, chat_id: str) -> int
    def cleanup_old(self, days: int = 90) -> int  # 90-day retention
```

### File 2: `robaivllm/thinking_proxy.py`

**Purpose:** FastAPI transparent proxy with thinking injection/extraction

```python
import os
import re
import json
import httpx
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from thinking_store import ThinkingStore

# Configuration
VLLM_BACKEND = os.getenv("VLLM_BACKEND_URL", "http://localhost:8077")
PROXY_PORT = int(os.getenv("THINKING_PROXY_PORT", "8078"))
DB_PATH = os.getenv("THINKING_DB_PATH", None)

app = FastAPI(title="Thinking Injection Proxy")
store = ThinkingStore(DB_PATH)
logger = logging.getLogger("thinking_proxy")

THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def extract_thinking(text: str) -> tuple[str, str]:
    """Extract <think>...</think> from text. Returns (thinking, cleaned_text)"""
    match = THINK_PATTERN.search(text)
    if match:
        thinking = match.group(1).strip()
        cleaned = THINK_PATTERN.sub('', text).strip()
        return thinking, cleaned
    return "", text


def inject_thinking_into_messages(chat_id: str, messages: list[dict]) -> list[dict]:
    """Prepend stored thinking to assistant messages based on message_id"""
    thinking_map = store.get_all_for_chat(chat_id)
    if not thinking_map:
        return messages

    result = []
    for msg in messages:
        if msg.get("role") == "assistant":
            msg_id = msg.get("id")
            if msg_id and msg_id in thinking_map:
                new_msg = msg.copy()
                thinking = thinking_map[msg_id]
                original_content = new_msg.get("content", "")
                new_msg["content"] = f"<think>\n{thinking}\n</think>\n\n{original_content}"
                result.append(new_msg)
                logger.debug(f"Injected thinking for message {msg_id}")
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy chat completions with thinking injection/extraction"""
    body = await request.json()

    # Extract metadata
    metadata = body.get("metadata", {})
    chat_id = metadata.get("chat_id")
    message_id = metadata.get("message_id")
    is_streaming = body.get("stream", False)

    # INBOUND: Inject thinking into historical assistant messages
    if chat_id and "messages" in body:
        body["messages"] = inject_thinking_into_messages(chat_id, body["messages"])
        logger.info(f"[INBOUND] Injected thinking for chat {chat_id}")

    # Forward to vLLM
    async with httpx.AsyncClient(timeout=300.0) as client:
        if is_streaming:
            return await handle_streaming(client, body, chat_id, message_id)
        else:
            return await handle_non_streaming(client, body, chat_id, message_id)


async def handle_streaming(client, body, chat_id, message_id):
    """Handle streaming response - pass through unchanged, copy thinking to DB"""
    accumulated_content = ""

    async def stream_generator():
        nonlocal accumulated_content

        async with client.stream(
            "POST",
            f"{VLLM_BACKEND}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"}
        ) as response:
            async for chunk in response.aiter_bytes():
                # Pass through chunk unchanged - client gets full response with <think> tags
                yield chunk

                # Accumulate content (copy) for thinking storage
                try:
                    chunk_str = chunk.decode('utf-8', errors='ignore')
                    for line in chunk_str.split('\n'):
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    accumulated_content += content
                            except json.JSONDecodeError:
                                pass
                except Exception:
                    pass

        # After stream complete - copy thinking content to DB (response already sent to client)
        if chat_id and message_id and accumulated_content:
            thinking, _ = extract_thinking(accumulated_content)
            if thinking:
                store.store(chat_id, message_id, thinking)
                logger.info(f"[COPY] Stored thinking for message {message_id} ({len(thinking)} chars)")

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


async def handle_non_streaming(client, body, chat_id, message_id):
    """Handle non-streaming response - return unchanged, copy thinking to DB"""
    response = await client.post(
        f"{VLLM_BACKEND}/v1/chat/completions",
        json=body,
        headers={"Content-Type": "application/json"}
    )

    result = response.json()

    # Copy thinking to DB (response returned unchanged to client)
    if chat_id and message_id:
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content:
            thinking, _ = extract_thinking(content)
            if thinking:
                store.store(chat_id, message_id, thinking)
                logger.info(f"[COPY] Stored thinking for message {message_id}")

    # Return full response unchanged - client gets <think> tags
    return result


# Passthrough for other endpoints (models, health, etc.)
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_passthrough(request: Request, path: str):
    """Pass through all other requests unchanged"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{VLLM_BACKEND}/{path}"

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

        return response.json() if response.headers.get("content-type", "").startswith("application/json") else response.content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
```

### File 3: `robaivllm/__init__.py`
Empty init file

### File 4: `robaivllm/requirements.txt`
**NOT NEEDED** - Dependencies come from mounted robaivenv

## Port Configuration Changes

**vLLM startup command must change:**
```bash
# Before: vLLM on 8078
# After: vLLM on 8077, Thinking Proxy on 8078

# vLLM (wherever it's configured - docker-compose, systemd, etc.)
--port 8077  # Changed from 8078

# Thinking Proxy
uvicorn thinking_proxy:app --host 0.0.0.0 --port 8078
```

**Environment variables:**
```bash
# In .env or wherever vLLM config lives
VLLM_BACKEND_URL=http://localhost:8077
THINKING_PROXY_PORT=8078
THINKING_DB_PATH=/path/to/robaivllm/data/thinking.db
```

## Deployment: Docker Container (using shared robaivenv)

### File: `robaivllm/Dockerfile`
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink so robaivenv python symlinks work
RUN ln -s /usr/local/bin/python3 /usr/bin/python3

# Copy application code
COPY robaivllm/*.py ./

# Create non-root user for security
RUN useradd -m -u 1000 proxyuser && \
    mkdir -p /app/data && \
    chown -R proxyuser:proxyuser /app

# Switch to non-root user
USER proxyuser

# Set environment variables
# venv will be mounted at /venv, prepend to PATH
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="/app:/robaitools"

# Run proxy server
CMD ["python3", "thinking_proxy.py"]
```

### Add to master `docker-compose.yml`
```yaml
  # Thinking Injection Proxy - transparent proxy for MiniMax M2 thinking preservation
  thinking-proxy:
    build:
      context: .
      dockerfile: robaivllm/Dockerfile
    container_name: thinking-proxy
    <<: [*restart-policy, *shared-environment]
    network_mode: host
    environment:
      - VLLM_BACKEND_URL=http://localhost:8077
      - THINKING_PROXY_PORT=8078
      - THINKING_RETENTION_DAYS=90
    volumes:
      - ./robaivenv:/venv:ro                        # Shared Python venv
      - ./robaimodeltools:/robaitools/robaimodeltools  # Shared library
      - ./robaivllm/data:/app/data                  # SQLite database
```

**Note:** User will handle changing vLLM port from 8078 → 8077 separately.

## Testing Plan

1. **Unit test thinking_store.py:**
   - Store/retrieve by message_id
   - Cleanup old records

2. **Integration test:**
   ```bash
   # Start vLLM on 8077
   # Start thinking proxy on 8078
   # Send multi-turn chat via curl
   curl -X POST http://localhost:8078/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Minimax-M2",
       "messages": [{"role": "user", "content": "Hello"}],
       "metadata": {"chat_id": "test-123", "message_id": "msg-001"},
       "stream": true
     }'
   ```

3. **Verify thinking in DB:**
   ```bash
   sqlite3 robaivllm/data/thinking.db "SELECT * FROM thinking_history;"
   ```

## Files to Create

| File | Action |
|------|--------|
| `robaivllm/__init__.py` | Create (empty) |
| `robaivllm/thinking_store.py` | Create |
| `robaivllm/thinking_proxy.py` | Create |
| `robaivllm/data/` | Create directory |
| `robaivllm/Dockerfile` | Create |
| `docker-compose.yml` | Add thinking-proxy service |

**No requirements.txt needed** - dependencies come from mounted robaivenv

## Configuration Changes Required

| Location | Change | Who |
|----------|--------|-----|
| vLLM startup | Port 8078 → 8077 | User handles |
| docker-compose.yml | Add thinking-proxy service | Me |

## Key Design Decisions

1. **Message-ID based storage**: Uses `message_id` from metadata to precisely align thinking with specific assistant responses, not just turn order.

2. **Transparent proxy**: All clients continue to use port 8078 unchanged - the proxy is invisible.

3. **Copy-through, not extract**: Responses pass through **unchanged** with `<think>` tags intact. We only **copy** the thinking content to SQLite for later re-injection. Clients (like Open WebUI) receive the full response and do their own stripping for display.

4. **INBOUND injection**: When a request comes in with conversation history, we prepend stored thinking to assistant messages so the model sees its previous reasoning.

5. **Self-contained**: All code lives in robaivllm/, no changes to robaiproxy needed.
