"""Validated limits shared by the CLI and runtime."""

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    cpu_threads: int = Field(default=1, ge=1)
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    mode: Literal["sequential", "static", "continuous"] = "continuous"
    scheduler: Literal["fcfs", "shortest_prompt"] = "fcfs"
    max_waiting: int = Field(default=64, ge=1)
    max_active: int = Field(default=8, ge=1)
    max_context: int = Field(default=2048, ge=2)
    max_output: int = Field(default=512, ge=1)
    max_batch_tokens: int = Field(default=16384, ge=2)
    max_kv_bytes: int | None = Field(default=None, ge=1)
    static_batch_wait_ms: float = Field(default=5, ge=0, le=1000)
    history_size: int = Field(default=2048, ge=1)
    local_files_only: bool = False

    @model_validator(mode="after")
    def valid_budget(self) -> "RuntimeConfig":
        if self.max_output >= self.max_context:
            raise ValueError("max_output must be smaller than max_context")
        return self


def load_config(path: str) -> RuntimeConfig:
    with open(path) as stream:
        return RuntimeConfig.model_validate(yaml.safe_load(stream) or {})
