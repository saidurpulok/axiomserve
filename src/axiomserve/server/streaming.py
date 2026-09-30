"""SSE framing and cancellation on client disconnect."""

import json
import time
from collections.abc import AsyncIterator

from starlette.requests import Request

from axiomserve.core.engine import Engine
from axiomserve.core.request import RequestState


def chunk(state: RequestState, model: str, text: str, reason: str | None) -> dict[str, object]:
    return {
        "id": state.request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": reason}],
    }


async def stream_tokens(
    state: RequestState, engine: Engine, request: Request, model: str
) -> AsyncIterator[str]:
    try:
        while True:
            event = await state.events.get()
            if await request.is_disconnected():
                engine.cancel(state.request_id)
                return
            payload = chunk(state, model, event.text, event.finish_reason)
            if event.error:
                payload["error"] = {"message": event.error}
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            if event.finish_reason is not None:
                yield "data: [DONE]\n\n"
                return
    finally:
        if not state.done.is_set():
            engine.cancel(state.request_id)
