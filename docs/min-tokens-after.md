# Minimum continuation after a natural boundary

```python
from rlstack import MinTokensAfter, SamplingSpec
from rlstack.runner.residents import EngineBuild

build = EngineBuild(enable_min_tokens_after=True)
sampling = SamplingSpec(
    max_tokens=8192,
    min_tokens_after=MinTokensAfter(sequence=(boundary_token_id,), min_tokens=1024),
)
```

Resolve the boundary IDs with the model's tokenizer and native template. The
option is model agnostic; it does not assume a particular thinking delimiter.
The first occurrence **in generated output** activates terminal suppression.
Before that occurrence, generation and natural termination are unconstrained.
After it, every resolved EOS/stop-token ID is suppressed until 1024 other raw
output IDs have been generated. Boundary IDs do not count; a later occurrence
of the boundary does count. Nothing forces the model to reach the boundary.

A floor on raw final IDs does not guarantee 1024 parsed teaching tokens, useful
content, valid wrappers or a completed answer. Late natural activation may
leave insufficient generation budget; retain that output as an unmet floor.
Static impossible budgets (including the model window) are rejected. No
extra request, forced closure, prompt injection or RNG reset is introduced.

The minimal implementation rejects string stops, repetition-based early
stopping, other minimum-token controls, forced thinking budgets, output
grammars and speculative decoding. They could otherwise bypass terminal
masking or conflict with it. Ordinary repetition/presence penalties remain
available. Requests without this option never allocate a constraint state;
old engine builds do not register the processor at all. Default build recipe
encoding and canonical bytes are preserved by leaving the capability unset.

`Turn.turn_extras['min_tokens_after']` contains the exact requested sequence,
minimum, resolved terminal IDs, processor name, budgets, first boundary end
(exclusive output index), raw after-boundary counts, activation/floor flags
and completion finish/stop reasons. Counts are reconstructed from returned
output IDs, not measurements of how often a GPU kernel applied a mask. The
normal Turn retains exact IDs, seed and bundle identity. An absent boundary,
a short final and a length-truncated output remain distinguishable.

For both floor-on and floor-off requests, the opted-in engine explicitly returns `raw_logprobs`, computed before masking,
penalties, temperature and top-p processing in vLLM 0.28. These are **not** the
constrained behavior-policy probabilities. `sampling_logprobs` extras record this limitation at engine scope;
existing GRPO importance ratios have not been qualified for this control. Use
frozen generation first. CPU tests establish counting and interface behavior;
GPU qualification through the shared Desk/Trainer remains required.

## Installed-source compatibility audit

Read-only audit of installed vLLM **0.28.0**, with no engine/model/GPU call:

- `v1/sample/logits_processor/interface.py`: BatchUpdate carries live generated
  output-ID lists; batch add/remove/move bookkeeping is part of the public API.
- `v1/sample/logits_processor/builtin.py`: the extension uses the same
  `process_dict_updates` helper as built-in processors.
- `v1/engine/input_processor.py`: resolves generation-config EOS IDs and the
  renderer's primary EOS through `SamplingParams.update_from_generation_config`.
  The wrapper resolves by the same path; admission verifies the exact set again.
- `v1/sample/sampler.py`: raw logprobs are computed before non-argmax-invariant
  processors. The custom processor masks terminals before greedy/random draws.
- `v1/sample/logits_processor/__init__.py`: custom processor names use
  `module:Class`; speculative decoding rejects custom processors.

The processor tests also ran on CPU torch using this installed interface and
its actual dictionary-update function loaded from source. No CUDA execution
was tested. Version-dependent accesses are explicit and fail rather than
silently falling back to an approximate minimum-total-token control.

Source SHA-256 fingerprints for that audit:

| File | SHA-256 |
| --- | --- |
| `interface.py` | `171dfc6ef9b5b53128e8eaecc4bdf1de413a706d64bf5be686048f583c66eee1` |
| `builtin.py` | `d8967b2358debb3ba34e0c5fc49b43c3a880e3d408d83f517575485701ad5cc9` |
| `sampler.py` | `f95805b2c6d06218c724826aa5f881a606e8d0a534a566aee7c85948520dd216` |

For a floor-off/on contrast, use the same opted-in engine for both arms.
This holds processor registration and raw-logprob configuration fixed.

Validation of this branch: 1,776 stdlib/fake tests passed (237 environment-gated
skips); the four processor tests additionally passed with CPU torch against
the installed 0.28 interface and dictionary-update source. Frozen GPU serving
qualification has not run.
