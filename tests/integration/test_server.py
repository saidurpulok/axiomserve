import json

from conftest import ControlledRunner
from fastapi.testclient import TestClient

from axiomserve.config import RuntimeConfig
from axiomserve.core.engine import Engine
from axiomserve.server.app import create_app


def test_streaming_and_completion_endpoints() -> None:
    config = RuntimeConfig(max_context=256, max_output=64)
    engine = Engine(ControlledRunner(), config)
    with TestClient(create_app(config, engine)) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 3})
        assert response.status_code == 200
        assert response.json()["choices"][0]["text"] == "xxx"
        assert response.json()["usage"]["completion_tokens"] == 3
        with client.stream(
            "POST", "/v1/completions", json={"prompt": "hi", "max_tokens": 3, "stream": True}
        ) as response:
            assert response.headers["x-request-id"]
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        assert "".join(c["choices"][0]["text"] for c in chunks) == "xxx"
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert client.get("/metrics").json()["counters"]["finished"] == 2
        assert client.delete("/v1/requests/unknown").status_code == 404
        assert (
            client.post("/v1/completions", json={"prompt": "", "max_tokens": 3}).status_code == 422
        )
        assert (
            client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 65}).status_code
            == 422
        )


async def test_http_cancellation_of_running_request() -> None:
    import asyncio

    import httpx

    config = RuntimeConfig(max_context=256, max_output=64)
    engine = Engine(ControlledRunner(0.01), config)
    app = create_app(config, engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(
                client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 64})
            )
            for _ in range(200):
                if engine.requests:
                    break
                await asyncio.sleep(0.001)
            request_id = next(iter(engine.requests))
            assert (await client.delete(f"/v1/requests/{request_id}")).status_code == 200
            response = await asyncio.wait_for(task, 3)
            assert response.json()["choices"][0]["finish_reason"] == "cancelled"
            assert not engine.requests


async def test_stream_generator_close_cancels_request() -> None:
    import asyncio

    from starlette.requests import Request

    from axiomserve.core.request import Status
    from axiomserve.core.sampling import SamplingParams
    from axiomserve.server.streaming import stream_tokens

    config = RuntimeConfig(max_context=256, max_output=64)
    engine = Engine(ControlledRunner(0.01), config)
    await engine.start()
    try:
        state = await engine.submit("hi", SamplingParams(max_tokens=64))

        async def receive() -> dict[str, object]:
            await asyncio.sleep(1)
            return {"type": "http.request", "body": b""}

        request = Request({"type": "http", "method": "POST", "path": "/"}, receive)
        stream = stream_tokens(state, engine, request, "tiny")
        assert (await anext(stream)).startswith("data: ")
        await stream.aclose()
        await asyncio.wait_for(state.done.wait(), 3)
        assert state.status == Status.CANCELLED
    finally:
        await engine.close()
