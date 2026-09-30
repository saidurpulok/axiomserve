from pydantic import Field

from axiomserve.core.sampling import SamplingParams


class CompletionBody(SamplingParams):
    prompt: str = Field(min_length=1, max_length=131072)
    stream: bool = False
