"""Thin HTTP transport over an injected inference engine."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from axiomserve.config import RuntimeConfig
from axiomserve.core.engine import Engine, OverloadedError
from axiomserve.core.request import Status
from axiomserve.core.sampling import SamplingParams
from axiomserve.runtime.model_runner import ModelRunner
from axiomserve.server.schemas import CompletionBody
from axiomserve.server.streaming import chunk, stream_tokens


def create_app(config: RuntimeConfig | None = None, engine: Engine | None = None) -> FastAPI:
    settings = config or RuntimeConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = engine
        if runtime is None:
            runner = await asyncio.to_thread(ModelRunner.load, settings)
            runtime = Engine(runner, settings)
        app.state.engine = runtime
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="AxiomServe", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        runtime: Engine = request.app.state.engine
        if not runtime.accepting:
            raise HTTPException(503, "inference worker is unavailable")
        return {
            "status": "ok",
            "model": settings.model,
            "mode": settings.mode,
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        runtime: Engine = request.app.state.engine
        return runtime.snapshot()

    @app.delete("/v1/requests/{request_id}")
    async def cancel(request_id: str, request: Request) -> dict[str, str]:
        runtime: Engine = request.app.state.engine
        if not runtime.cancel(request_id):
            raise HTTPException(404, "request is unknown or already terminal")
        return {"request_id": request_id, "status": "cancellation_requested"}

    @app.post("/v1/completions")
    async def completions(body: CompletionBody, request: Request) -> Any:
        runtime: Engine = request.app.state.engine
        params = SamplingParams.model_validate(body.model_dump(exclude={"prompt", "stream"}))
        try:
            state = await runtime.submit(body.prompt, params)
        except OverloadedError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        if body.stream:
            return StreamingResponse(
                stream_tokens(state, runtime, request, settings.model),
                media_type="text/event-stream",
                headers={"X-Request-ID": state.request_id, "Cache-Control": "no-cache"},
            )
        try:
            while not state.done.is_set():
                if await request.is_disconnected():
                    runtime.cancel(state.request_id)
                    raise HTTPException(499, "client disconnected")
                try:
                    await asyncio.wait_for(state.done.wait(), timeout=0.1)
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            runtime.cancel(state.request_id)
            raise
        if state.status == Status.FAILED:
            raise HTTPException(500, state.error)
        response = chunk(state, settings.model, state.output_text, state.finish_reason)
        response["usage"] = {
            "prompt_tokens": len(state.input_ids),
            "completion_tokens": len(state.generated_ids),
            "total_tokens": len(state.input_ids) + len(state.generated_ids),
        }
        return response

    return app
