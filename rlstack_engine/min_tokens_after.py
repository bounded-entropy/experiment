"""Opt-in vLLM 0.28 batch processor; no effect before natural activation.

This is a sampling constraint, not an RL importance-ratio implementation.
Returned raw model logprobs are not constrained behavior-policy logprobs.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig

import torch
from vllm.v1.sample.logits_processor import (
    BatchUpdate, LogitsProcessor, process_dict_updates,
)

from rlstack.sampling import CONTINUATION_KEY, ContinuationCounter
from rlstack.spec.specs import MinTokensAfter


@dataclass
class RequestState:
    counter: ContinuationCounter
    output_ids: list[int]
    minimum: int
    consumed: int = 0


class MinTokensAfterProcessor(LogitsProcessor):
    def __init__(self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool) -> None:
        if vllm_config.speculative_config is not None:
            raise ValueError("min_tokens_after does not support speculative decoding")
        self.requests: dict[int, RequestState] = {}

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        config = (params.extra_args or {}).get(CONTINUATION_KEY)
        if config is None:
            return
        floor = MinTokensAfter(tuple(config["sequence"]), config["min_tokens"])
        if params.max_tokens is None or params.max_tokens < len(floor.sequence) + floor.min_tokens:
            raise ValueError("max_tokens cannot accommodate min_tokens_after")
        if params.stop:
            raise ValueError("min_tokens_after cannot gate string stops")
        if params.repetition_detection is not None:
            raise ValueError("min_tokens_after cannot gate repetition early stopping")
        if params.ignore_eos or params.min_tokens:
            raise ValueError("min_tokens_after requires ordinary EOS and min_tokens=0")
        if params.thinking_token_budget is not None:
            raise ValueError("min_tokens_after requires a natural activation boundary")
        if params.allowed_token_ids is not None or params.structured_outputs is not None:
            raise ValueError("min_tokens_after cannot combine with constrained output grammars")
        terminals = config["terminal_token_ids"]
        if not terminals or any(type(t) is not int or t < 0 for t in terminals):
            raise ValueError("min_tokens_after requires nonnegative terminal token IDs")
        if set(floor.sequence) & set(terminals):
            raise ValueError("activation sequence cannot contain terminal token IDs")

    @staticmethod
    def _new_request(params: SamplingParams, prompt_ids: list[int] | None,
                     output_ids: list[int]) -> RequestState | None:
        config = (params.extra_args or {}).get(CONTINUATION_KEY)
        if config is None:
            return None
        MinTokensAfterProcessor.validate_params(params)
        terminals = tuple(config["terminal_token_ids"])
        if set(terminals) != set(params.all_stop_token_ids):
            raise ValueError("resolved terminal IDs changed before min_tokens_after enforcement")
        return RequestState(ContinuationCounter(tuple(config["sequence"]), terminals),
                            output_ids, config["min_tokens"])

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.requests, batch_update, self._new_request)

    def is_argmax_invariant(self) -> bool:
        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, state in self.requests.items():
            if len(state.output_ids) < state.consumed:
                raise ValueError("min_tokens_after requires append-only generated token IDs")
            state.counter.extend(state.output_ids[state.consumed:])
            state.consumed = len(state.output_ids)
            if state.counter.suppresses_terminals(state.minimum):
                terminals = tuple(state.counter.terminals)
                if any(t >= logits.shape[-1] for t in terminals):
                    raise ValueError("resolved terminal token ID is outside vocabulary")
                logits[index, list(terminals)] = -torch.inf
        return logits
