"""Small command surface for serving and reproducible benchmark sweeps."""

import argparse
import asyncio
import json
import logging
from typing import Any

import uvicorn
import yaml

from axiomserve.bench.reporting import save_report
from axiomserve.bench.runner import benchmark
from axiomserve.bench.workload import BenchConfig
from axiomserve.config import RuntimeConfig, load_config
from axiomserve.server.app import create_app


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in ["request_id", "status", "reason", "phase"]:
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="axiom")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config")
    serve.add_argument("--model")
    serve.add_argument("--mode", choices=["sequential", "static", "continuous"])
    serve.add_argument("--device", choices=["cpu", "cuda", "auto"])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    bench = subparsers.add_parser("bench")
    bench.add_argument("--config", default="configs/bench.yaml")
    bench.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    if args.command == "serve":
        settings = load_config(args.config) if args.config else RuntimeConfig()
        updates = {
            name: getattr(args, name)
            for name in ["model", "mode", "device"]
            if getattr(args, name) is not None
        }
        settings = RuntimeConfig.model_validate(settings.model_dump() | updates)
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
    else:
        with open(args.config) as stream:
            settings_bench = BenchConfig.model_validate(yaml.safe_load(stream))
        report = asyncio.run(benchmark(settings_bench))
        save_report(report, args.output)
        print(f"Saved {len(report['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
