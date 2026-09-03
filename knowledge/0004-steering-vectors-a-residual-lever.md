# ADR 0004 — A steering vector is an adapter type on a residual lever, served by a hook in the engine image

| | |
|---|---|
| **Date** | 2026-09-02 |
| **Status** | Implemented (2026-09-03; CONTEXT #77) — Accepted 2026-09-02: Q1, Q3, Q4, Q6, Q7, Q9, Q10 agreed; Q5 and Q11 resolved by the agent at Samarth's delegation; Q8 deferred to a later ADR, its question answered below; Q2 REFOLDED — positions are a per-request directive, recorded and salted — and Q2a agreed, with "steer constantly on decode" folded as the default |
| **Author** | Claude Fable 5.1 (session: the vLLM-Hook assessment, 2026-09-02) |
| **Touches** | `policy/adapters/` (one new adapter type, three files: `steer.py`, `steer_torch.py`, `steer_vllm.py`), `policy/adapters/base.py` (one `Mechanism` member), `policy/adapters/rollout.py` (two `Levers` fields), `runner/engines/vllm_engine.py` (the bus pays the two new levers, and builds `Request` with `occupied` and the directives), `client.py` + `runner/interfaces.py` + `runner/traffic.py` + `runner/remote.py` (Q2: one keyword, `directives`, on `sample` and `score`, carried over the wire), `runner/fakes.py` (the fake engine's inventory and the keyword), `rlstack_engine/` (the first real plugin: `steer.py`, and `BatchView.from_vllm` built), `spec/validate.py` (one check, Q7), `spec/specs.py` (sugar), `deploy/steer_l4.py` (the metal proof, through the desk), `tests/` |
| **Invariants** | I2 (an adapter type ships both lowerings — this one's rollout half is the first on a plugin), I7 (the plugin probes at boot and refuses the build it cannot serve), I8 (per-request selection inside one fused batch, on a lever the engine does not natively index), I12 (the check ends by the desk's `release`, never the venue's timer) |
| **CONTEXT** | extends #29 (the plugin contract: probe / slots / `cache_salt` / BatchView), #46 (soft prompts served, side attention refused — and the reason it was refused), #48 (the Lowering: one contract per adapter type per side), #76 (ADR 0003's `release`, whose Modal half is still UNPROVEN and which this ADR's check observes); the entry number lands at implementation |

## Original prompt

> could you look into vLLM hook? it's a recent library that got implemented
> which can actually implement hooks into vLLM for steering vectors, etc. I
> feel like if I implemented it along with multi-tenant in my repo, it could
> interface very nicely. what do you think?

Then, after the assessment:

> before you draft the ADR, just making sure, would you be able to insert a
> custom residual at any part of the model? would it be mergable easily, like
> adding two different residuals at two different layers (would be two
> different adapters in this case)? if we follow site schema, we can declare a
> range of layers to apply the adapter onto, etc. does this all work out?

And the ask:

> ok nice. could you create an ADR for steering vectors? the check should be a
> real deployment onto modal (along with a takedown of the metal, by calling
> the desk (since we have this implemented.)

And, answering in session (2026-09-02):

> Q1) agree
>
> Q2) ideally id be able to modify which positions at runtime. this is
> interesting -- i guess it's a property of the request itself.
> architecturally, adding some sort of meta information parameter to the
> engine adapter add-on or whatever doesnt seem like too big of a deal. what
> do you think (i dont see any issues). what do you think?
>
> Q3) i agree, we should salt the cache with the bundle id obfviously.
>
> 4) i agree with your suggestion.
>
> 5) decide what you think is good
>
> 6) sure
>
> 7) yea wait this should already be the case.
>
> 8) yea let's do that in a later ADR. quick question: with this new
> self_attn output, will that remove the need for the custom flash attention
> rewrite for the logit bias?
>
> 9) yea this makes sense to m
>
> 10) sure. id like to ideally test this under different conditions (TP,
> etc.) shortly. just make sure at the end of tests, all metals are
> deallocated.
>
> 11) just decide what's best for these races.

## Context / problem

**What is true today.** The serving levers are a CLOSED set
(`policy/adapters/base.py:33-47`): `punica`, `prompt_embeds`, `logits`,
`side_attention`, `none`. Three of them reach a site: punica reaches weighted
projections, prompt_embeds reaches the embedding table, and `logits` is
declared and used by no adapter type. Nothing reaches a residual BOUNDARY —
`resid_pre.<n>` at `model.layers.<n>`, `final_hidden` at `model.norm` — on the
engine side, although the schema has carried those sites since #15
(`policy/siteschema.py:148`, `:195`) and the replay side taps one today: the
value head's `VhSite` wraps the boundary module and captures its output
(`policy/adapters/value_head_torch.py:99-108`), a replay lowering with no
rollout twin (`serving = None`).

The lever that WOULD reach a boundary — a plugin — has a written contract and
no instance. `rlstack_engine/plugin.py` names what a plugin must re-earn
(probe at boot, per-slot banks, `cache_salt`, a per-forward verb written
against `BatchView`), and `BatchView.from_vllm` is unbuilt on purpose
(`rlstack_engine/batch_view.py:43`): "a shim written against metadata no
plugin exercises would be a guess pinned to a version". The one plugin
designed, side attention, stalled at the KERNEL (#46: the dense
FlashAttention path hands back no LSE; the CORRECTION at `CONTEXT.md:1415`
found a 159-line shim but nothing merged; reachability is NONE). So "an
engine plugin" has, in this repo's experience, meant "a kernel seam that is
not there".

**What the outside just proved.** Two libraries, both current in 2026, hook
vLLM's residual stream on the V1 runner without touching a kernel:

- IBM's vLLM-Hook (arXiv 2603.06588; ICML 2026 expo; Apache-2.0). A worker
  extension on the GPU worker, `register_forward_hook` on the decoder layers,
  per-request configuration through `SamplingParams.extra_args["steer"]`, and
  the batch sliced per request off `get_forward_context().attn_metadata`'s
  `query_start_loc` plus the runner's request ids. The steering worker is 186
  lines. It steers only the LAST token of each request's slice
  (`target = slice_view[-1:]`), loads the vector from a file path and caches
  it per worker, guards `torch.cuda.is_current_stream_capturing()`, forces
  `enforce_eager=True` by default, and says nothing about the prefix cache or
  tensor parallel.
- UK AISI's vllm-lens (MIT; `vllm>=0.16.0`; v1.2.1, 2026-07-22). A general
  plugin by entry point that patches `EngineArgs.create_engine_config` to
  force its worker extension, eager mode and the V1 model runner for EVERY
  engine in the image; pre- and post-hooks on every decoder layer; vectors
  delivered by `collective_rpc("set_steering_data", (request_id, pickle))`
  before generate and cleared after; applied at every position, prefill and
  decode, with `scale`, `norm_match` and `position_indices`; per-request
  slices off `query_start_loc`; and `skip_reading_prefix_cache=True` set on
  every steered request. On the fused-residual layers vLLM uses it adds to
  `output[0]` and takes `output[0] + output[1]` as the stream for norm
  matching.
- Upstream, vLLM RFC 36998 (2026-03-13) proposes a native observation plugin,
  prefill-only; open, no PR.

What they establish for us, exactly: (a) a decoder layer's forward is a seam
vLLM keeps stable across model families, (b) the token → request map under
continuous batching is `query_start_loc` + request ids, which IS the
`token_slot` gather `BatchView` was written for, and (c) the price is eager
mode. What they do NOT do: speak bundles (both key steering by request id or
file path), salt the prefix cache for tenancy (one ignores it, the other
disables reads), or pay only for what a build serves (vllm-lens patches the
engine globally). So this ADR ports the sixty lines that matter and takes no
dependency.

**The bank rule is documented, not enforced.** `replay.py:117` states "a site
path carries at most one delta PER TENANT (the bank rule)". No Phase-0 check
says so: run in this session, `validate()` on a bank with `lora` at
`layers.0-3.self_attn.q_proj` and a second `lora` at
`layers.2.self_attn.q_proj` returns no issues. Both states install at layer 2
(`SiteWrapper.add` refuses only the SAME state object) and both apply to the
tenant's rows — silently summed, for every adapter type. "Mergeable" in the
prompt's sense depends on this rule being real (Q7).

**What already costs nothing.** `VllmEngine` defaults `enforce_eager=True`
(`runner/engines/vllm_engine.py:52`) and so does every resident
(`runner/residents.py:84`); the soft prompt already gives up the V2 model
runner (`soft_prompt_vllm.py:44`). The biggest cost the two libraries pay is
already paid on this fleet's builds.

**The desk's release is unproven on the venue.** ADR 0003 landed `release`
and `reacquire` (CONTEXT #76) with the explicit non-promise "No
release-then-knock has been seen on Modal — whether the shift's return gets
the container reclaimed". The check this ADR asks for — a real Modal
deployment ended by CALLING THE DESK — is the first time that is observed.

*Measured:* the fakes suite is 835 tests locally (95 torch-gated skips).
*Unmeasured:* the throughput cost of hooks on an eager L4; no steer exists to
measure.

## Decision

**A steering vector is an adapter type, `steer`, on a new lever,
`Mechanism.RESIDUAL`: a per-slot vector the engine image ADDS to the residual
stream at a boundary site, selected per request, salted into the prefix cache
by bundle.** One bank entry is one vector of width `d` per matched boundary
(`resid_pre.<n>`, `final_hidden`; a range like `resid_pre.8-20` is one entry
with one vector per layer, `tie=True` shares one across the range — the LoRA
shape, `lora_torch.py:120`); zero is its exact identity, so init is zero by
default and the gradient is non-zero from the first step. The replay lowering
is a `SiteWrapper` at the boundary path that adds each routed row's vector to
the module's output — at every position by default, or inside the WINDOW
that row's turns recorded — and is transparent to every other tenant's rows:
the value head's tap plus one add. WHICH positions is a property of the
request (Q2, refolded): an environment may pass a `SteerWindow` directive
with `sample`, the rollout lowering resolves it against the request it sees
(real-token coordinates, offset by the positions other adapter types occupy)
and RECORDS the resolved window as a turn fact (I6), the replay wrapper reads
it back from `ReplayRows.facts` and masks the add to those positions, and a
trajectory whose turns disagree is refused at replay — plora's one-draw rule
(`plora_torch.py:344`). The rollout lowering demands a worker class and eager
mode, attaches a bundle's vectors as ONE content-addressed file under the
build's workdir (what `lora_vllm.attach` does with a PEFT dir,
`lora_vllm.py:55-73`), applies by naming that file and the resolved window in
the request's `SamplingParams.extra_args` and the bundle PLUS the window in
its `cache_salt`, aligns zero positions, and detaches by removing the file. The engine-image
half, `rlstack_engine/steer.py`, is the first real `EnginePlugin`: it probes
four symbols at boot, installs one hook per served boundary at model load,
builds `BatchView.from_vllm` off the forward context, holds a bounded
per-worker bank keyed by file (loaded on first sight — the `LoRARequest`
precedent), and adds `bank[slot]` to each token's row. Two entries at two
layers are two adapters that never meet; two entries at one layer in one bank
are refused at Phase 0 (Q7). The proof is `deploy/steer_l4.py`: a desk and
one L4 metal on Modal, two tenants (lora-only, steer-only) submitted THROUGH
THE DESK onto one serving host and one learner, a zero-tolerance parity
control, and the metal handed back by `RemoteDesk.release` with the
container's reclaim observed.

### Touched / untouched

- **Touched** — `policy/adapters/base.py`: `Mechanism.RESIDUAL = "residual"`,
  one line and its docstring ("a vector added at a boundary by the engine
  image; ours, shipped as a plugin"). The set stays closed; it gains a member.
- **Touched** — `policy/adapters/steer.py` (declaration: `serving =
  RESIDUAL`, `site_ok` = the value head's predicate — boundary, unweighted —
  `init = {d, tie, init_std, seed}`, both lowerings entered lazily, rule 7),
  `steer_torch.py` (replay: `SteerState`, `SteerSite(SiteWrapper)`, build /
  install / uninstall / emit / load / `merge_vectors`), `steer_vllm.py`
  (rollout: `SteerRollout(RolloutLowering)`, five verbs plus detach). One file
  per adapter type per side — rule 8's `adapters/` line, exactly.
- **Touched** — `policy/adapters/rollout.py`: `Levers` grows `extra_args`
  (per-request selection the worker reads) and `cache_salt` (prefix-cache
  identity the engine cannot hash itself); `merged_with` joins the first and
  refuses two different salts; `claims` may name either. `Request` grows
  `occupied` (the positions other adapter types put in front of the real
  tokens, summed by the bus — what a window is offset by) and `directives`
  (the typed per-request records the caller passed; an adapter type picks
  its own by type). `check_levers_compose` unchanged.
- **Touched** — `client.py`, `runner/interfaces.py`, `runner/traffic.py`,
  `runner/remote.py` (Q2): `PoolClient.sample` and `score` gain one keyword,
  `directives: Sequence[Directive] = ()`; `Engine.sample_tokens` /
  `score_tokens` carry it; the wire frame encodes each directive by its
  adapter type's name and the adapter type decodes its own record. An
  environment that passes nothing is unchanged — every call site today
  passes nothing (`math_single_turn.py:14`, `dapo_math.py:33`).
- **Touched** — `runner/engines/vllm_engine.py`: the bus pays the two levers
  — `sample_tokens` and `score_tokens` fold `levers.extra_args` into the
  `SamplingParams` they build (`:222`, `:281`) and `levers.cache_salt` into
  whatever prompt form `_levers_for` produced — and builds `Request` with
  `occupied` and the directives. No mechanism word enters the file;
  `_pay_demands` still refuses a `plugin=` demand — this plugin arrives as
  ENGINE ARGS (Q5), which the bus already pays.
- **Touched** — `runner/fakes.py`: `FakeEngine.reachability` answers
  `RESIDUAL` for boundary sites when its build serves `steer`, so Phase 0 and
  the resume suite exercise the adapter type without torch.
- **Touched** — `rlstack_engine/steer.py` (the plugin: `SteerWorker`, the
  hook, the per-worker bank, the probe), `rlstack_engine/batch_view.py`
  (`from_vllm` built, the one version-pinned shim), `rlstack_engine/plugin.py`
  (Q9: the per-forward verb becomes the mechanism's own).
- **Touched** — `spec/validate.py`: `check_sites_do_not_overlap` (Q7);
  `spec/specs.py`: `steer(site, d, tie=False, init_std=0.0)` sugar beside
  `lora()`.
- **Touched** — `deploy/steer_l4.py`: the check (Q10). `deploy/` stays venue
  only (I5): the spec, the bank, the loss are declared there and nothing
  semantics-bearing lives there.
- **Touched** — `ARCHITECTURE.md` (Mechanism / lever gains the member; the
  "why the set is closed" blockquote gains its first tier-three plugin that
  EXISTS; `resid_pre.<n>`'s tensor stated, Q1), `STYLE.md` untouched,
  `rl-stack-spec.md` at the v4 fold (I2's mechanism list), `CONTEXT.md` at
  implementation.
- **Untouched** — `runner/learners/torch_learner.py`, `fsdp_torch.py`: the
  replay lowering is a site wrapper routed by the row plan, the #44 shape,
  and the recorded window reaches it through `ReplayRows.facts`, which the
  learner already threads adapter-blind (`torch_learner.py:349`); no learner
  file changes, as #46 and #48 both managed. `policy/adapters/replay.py`:
  `SiteWrapper`, `join_site`, `RowPlan`, `facts` are used as they stand.
  `data/trajectory.py`, `data/flatten.py`: `Turn.turn_extras` and
  `TokenBatch.doc_turn_extras` already carry a per-turn fact to the row.
- **Untouched** — `policy/siteschema.py`: no new sites (Q8); the grammar
  already has numeric ranges (`:45-65`). `policy/compile.py`: a steer payload
  is a payload; `group_by_adapter_type` routes it. `runner/residency.py`:
  bounded residency and `pinned()` apply unchanged. `runner/restore.py`: a
  steer bundle rebuilds from the store like any other.
- **Untouched** — `runner/desk.py`, `runner/remote.py`, `runner/host.py`,
  `runner/residents.py`: the check USES `release`; it changes nothing on the
  desk. `EngineBuild.serves` already carries adapter types and
  `enforce_eager` already exists (`residents.py:77-85`).
- **Untouched** — `spec/canonical.py`, identity: a spec without a steer entry
  hashes as before; one with it hashes the adapter type's code through the
  registry as every adapter type does. `observe/`: a steer entry is a bank
  entry in `dictionary.json` and its rail is the existing `logprob_gap`; the
  fleet journal's `release` event already renders.
- **Untouched** — `rlstack_engine/side_attention.py`, `certificates.py`,
  `slots.py`: side attention stays refused for its own reason; `SlotTable` is
  used by the worker's bank as written.

### Promises / non-promises

- **Promises — the gate.** A spec with `steer("resid_pre.8-20", d=1024)` in
  its bank passes Phase 0 against a build whose `serves` names `steer`, is
  refused `site-unreachable` (reachability NONE) against one that does not,
  and a build asked to serve `steer` that cannot pay — graph capture on, a
  model runner the shim does not know — refuses AT CONSTRUCTION with the
  missing seam named (`ProbeError`, I7), never mid-run. Two entries of ANY
  adapter type resolving to one site in one bank are refused
  `site-overlap` (Q7).
- **Promises — the fakes.** The suite is green; `tests/test_resume.py` is
  unchanged and green, and gains one case with a steer entry whose run dir
  is byte-identical across resume; every existing golden run id is
  unchanged. The plugin lifecycle is pinned in `tests/test_engine_plugin.py`
  the way side attention's is: probe names the missing seam, the bank is
  per-slot, a token gathers from its own request's slot, an unknown slot
  raises.
- **Promises — the window (Q2).** A `SteerWindow` passed with `sample` is
  applied on the engine to exactly the positions it names, is sealed into
  `Turn.turn_extras` resolved (slice coordinates, offset included), and the
  replay wrapper adds at exactly those row positions and nowhere else —
  pinned in the fakes by a round trip (directive → turn → batch → mask) and
  on metal by Promise 2 run under a window. A call with no directive steers
  every position, which is the recorded default. A trajectory whose turns
  recorded different windows is refused at replay with the row named. The
  prefix-cache salt carries the window beside the bundle id, so two requests
  under one bundle and two windows never share a block.
- **Promises — on metal, `deploy/steer_l4.py`** (Q10):
  1. **Zero-tolerance control.** With `v = 0`, the served per-token logprobs
     and the replayed ones are BIT-IDENTICAL to the base's (max |Δ| =
     0.00e+00 on both sides) — the #46 control, in the steer's shape: the hook
     and the wrapper change no number when the vector is the identity.
  2. **Parity.** With `v ≠ 0` (a seeded `init_std`), served vs replayed
     per-token logprobs agree within the bf16 budget the `logprob_gap` rail
     already runs under, reported as a number.
  3. **Tenancy through the desk.** Two specs — `lora` only, `steer` only —
     submitted by `RemoteDesk.submit`, the second JOINing the first's
     listings (one serving host, one learner, on one L4), two updates each,
     prefix caching ON; and a prefix shared by both tenants scores under each
     bundle exactly what a cache-cold serial run gives — no aliasing across
     bundles, reuse within one.
  4. **Takedown by the desk.** `RemoteDesk.release(metal)` answers
     `told: True`; `status()` reports the metal `released` and its listings
     gone; the spawned keepalive input RETURNS — `until_released` fired, the
     container reclaimed because the desk decided (ADR 0003 Q3, seen for the
     first time).
  5. **No metal left standing (Q10).** Every entrypoint that acquires metal
     ends — in a `finally`, whatever the run did — by asking the desk to
     release every metal it holds and then ASSERTING, from `status()`, that
     no metal is on the plane (`plane: False` for every row). A `sweep`
     entrypoint does the same for a run that died before its `finally`. The
     venue's scaledown stays the backstop, set no shorter than the desk's
     limit (ADR 0003 Q3).
- **Non-promises.** Tensor parallel above 1 is UNPROVEN (the add is
  replicated per rank by construction; not measured) — but the deploy takes
  `--tp` and `--gpu`, so the follow-up Samarth asked for (TP and other
  conditions, "shortly") is one flag and a second L4, not a new script.
  Pipeline parallel,
  CUDA graphs (forfeited by demand), norm matching and positional subsets
  (not built — Q2), mid-layer boundary sites (Q8), the knock back after
  release unless Q10's last step runs green, and the throughput cost of eager
  plus hooks — measured by the meter and REPORTED in the outcome, not
  promised. A parity certificate stays unwired, as it is for every adapter
  type (I2's caveat).

### Interfaces

`Mechanism` gains `RESIDUAL`. `Levers` gains `extra_args` and `cache_salt`
and the bus pays both in the two request paths it owns. `Request` gains
`occupied` and `directives`; `PoolClient.sample` / `score`,
`Engine.sample_tokens` / `score_tokens` and the wire's two frames gain
`directives`, encoded per adapter type. The steer adapter type declares
`records = ("steer_window",)` — the per-turn fact the rollout writes and
the replay reads (`ReplayRows.facts`). `BuildDemands`
carries the plugin as ENGINE ARGS (`worker_cls`, `enforce_eager`), so a build
that serves `steer` is `VllmEngine(serves=("lora", "steer"))` and nothing
else changes at construction. `Engine.reachability` answers `RESIDUAL` for
`model.layers.<n>` and `model.norm` on such a build. The plugin's
`required_symbols` are the forward context, the attention metadata's
`query_start_loc`, the runner's request table, and `SamplingParams.extra_args`
— probed on the worker at model load. The Phase-0 gate grows one code,
`site-overlap`. A steer payload is one safetensors blob per bank entry
(`adapters/<name>@<v>.safetensors`, as every adapter type emits). The store
learns nothing new; `observe/` sees a bank entry and the existing rail. The
metal check's door is `RemoteDesk.release` (`runner/remote.py:812`), the
verb ADR 0003 added and called "also a manual door".

### Sketches

```python
# policy/adapters/base.py
class Mechanism(StrEnum):
    ...
    RESIDUAL = "residual"     # a per-slot vector added at a boundary by the engine image (ours)


# policy/adapters/steer.py
@adapter_type("steer")
class Steer(AdapterType):
    serving = Mechanism.RESIDUAL

    def site_ok(self, meta: SiteMeta) -> bool:
        """A residual boundary: unweighted and boundary — the value head's
        predicate. Which boundaries a BUILD reaches is the engine's answer."""
        return meta.is_boundary and not meta.has_weight
    # init: d (width), tie=False, init_std=0.0 (zero IS the identity), seed


# policy/adapters/steer.py — the directive (Q2): a typed per-request record
@dataclass(frozen=True)
class SteerWindow:
    """Which positions of THIS request the steer applies to, in real-token
    coordinates: 0 is the first prompt token, len(prompt) the first generated
    one, None runs to the end of generation. Absent: every position."""
    start: int = 0
    end: int | None = None


# policy/adapters/steer_torch.py
class SteerSite(SiteWrapper):
    """The boundary add: pass the module's output through with each routed
    row's vector added inside the row's RECORDED window (every position when
    none was recorded), and every other row untouched."""
    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        rows = self.plan.rows
        for state in self.installed:
            mask = window_mask(rows, state.slot, out.shape[1])   # [rows, W, 1] off facts; raises if a row's turns disagree
            out = out + mask * state.rows_delta(self.path, rows)  # [rows, 1, d], zero off-slot
        return out


# policy/adapters/rollout.py
@dataclass(frozen=True)
class Request:
    token_ids: tuple[int, ...]
    seed: int | None = None
    occupied: int = 0                              # positions other adapter types put in front
    directives: tuple[Any, ...] = ()               # the caller's typed records; pick yours by type


# policy/adapters/steer_vllm.py
class SteerRollout(RolloutLowering):
    adapter_type = "steer"
    mechanism = Mechanism.RESIDUAL
    claims = ("extra_args.rlstack_steer", "cache_salt")
    # apply(): window = the SteerWindow among request.directives, or the default;
    # resolved = (occupied + start, occupied + end) in slice coordinates;
    # extra_args carries the file AND the resolved window; turn_extras records
    # {"steer_window": [start, end, occupied]}; cache_salt = f"{bundle}/{start}:{end}"

    def demands(self) -> BuildDemands:
        return BuildDemands(engine_args={
            "worker_cls": "rlstack_engine.steer.SteerWorker",   # Q5
            "enforce_eager": True})                              # Q6

    def reaches(self, meta: SiteMeta) -> bool:
        return meta.path.startswith("model.layers.") or meta.path == "model.norm"

    def attach(self, bundle_id: str, payloads: Mapping[str, bytes]) -> Path:
        """The bank's steer entries fused into ONE file — {path: vector} in
        the served dtype, width-checked against hidden_size — under
        workdir/<bundle_id>/steer.safetensors. The file is the address."""

    def apply(self, attached: Path, request: Request) -> Levers:
        return Levers(extra_args={"rlstack_steer": str(attached)},
                      cache_salt=attached.parent.name)             # the bundle id (Q3)

    def detach(self, attached: Path) -> None: ...                  # rmtree, as lora_vllm


# policy/adapters/rollout.py
@dataclass(frozen=True)
class Levers:
    prompt: Any | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    turn_extras: Mapping[str, Any] = field(default_factory=dict)
    extra_args: Mapping[str, Any] = field(default_factory=dict)   # SamplingParams.extra_args
    cache_salt: str | None = None                                  # prefix-cache identity, per bundle


# rlstack_engine/steer.py — ships in the engine image, named by string
class SteerWorker(Worker):                       # vllm.v1.worker.gpu_worker.Worker (Q5)
    def load_model(self) -> None:
        super().load_model()
        probe(self)                              # I7: the four seams, or ProbeError at boot
        self.bank = SteerBank(max_slots=...)     # per-worker, keyed by file, bounded (Q4)
        for layer_idx, module in boundaries_of(self.model_runner.model):
            module.register_forward_hook(self._add_at(layer_idx))

    def _add_at(self, layer_idx):
        def hook(module, args, output):
            view = BatchView.from_vllm(self.model_runner, layer_idx, self.bank.slot_of)
            hidden = output[0] if isinstance(output, tuple) else output      # fused pair (Q1)
            hidden.add_(self.bank.gather(view, layer_idx))                  # [tokens, d]
            return output
        return hook


# rlstack_engine/batch_view.py
@staticmethod
def from_vllm(runner, layer_idx: int, slot_of: Callable[[str], int]) -> BatchView:
    """query_start_loc + the runner's req_ids -> one slot per token, read off
    each request's extra_args["rlstack_steer"]; a request with none is slot -1
    (adds nothing); a named file the bank does not hold RAISES (Q11)."""
```

## Questions

**Q1. Which tensor does `resid_pre.<n>` name, and where does vLLM's fused
pair get the add?** The site's path is `model.layers.<n>` and the replay
wrapper taps that module's OUTPUT — the stream LEAVING layer n — which the
name calls "pre". vLLM's layer returns `(hidden, residual)` whose sum is the
stream; the next fused norm and the final norm both sum them.
Recommendation: **the site names the module's output on both sides, exactly
as the value head reads it today; the vLLM hook adds to `hidden`
(`output[0]`), vllm-lens's choice; `final_hidden` is `model.norm`'s output on
both sides. The name is not changed** — `AdapterSpec.site` strings hash into
run identity, so a rename is a separate identity-breaking commit if ever —
and the meaning is written once, in `siteschema.py`'s docstring and
ARCHITECTURE.md. Adding to `hidden` rather than to the materialized sum
differs by bf16 rounding only; that is the parity budget, and Promise 2
measures it.
If the other branch: the steer adds at the layer's INPUT (a pre-hook, so the
name is exact) and the value head keeps its output tap — two adapter types
reading one site name as two tensors, which is the mismatch this repo exists
to refuse.

> **Samarth:** agree — "Q1) agree"

**Q2. Which positions?** IBM steers the last token of each slice per forward
(the last prompt position, then every decode token); vllm-lens steers every
position; both allow position subsets.
Recommendation: **every real position — prompt, completion, and a soft
prompt's virtual rows — at prefill and decode, no scale, no norm matching.**
It is the one rule a replay lowering reproduces with nothing but the row
plan (padding positions are causally after every real one and outside the
loss, so the wrapper adds everywhere and masks nothing); a learned vector
has no need of a separate scale; norm matching rescales by a data-dependent
bf16 norm the replay forward would have to match bit for bit.
If the other branch: completion-only or last-token needs the prompt length
per row on the replay side (the boundary catches the attention mask, as the
value head does) and, for last-token, a rule for the prefill's final position
that scoring traffic (one prefill, no decode) breaks.

> **Samarth:** disagree, extended — "ideally id be able to modify which
> positions at runtime. this is interesting -- i guess it's a property of the
> request itself. architecturally, adding some sort of meta information
> parameter to the engine adapter add-on or whatever doesnt seem like too big
> of a deal. what do you think (i dont see any issues)."

**Refolded.** Agreed: the positions are a property of the REQUEST, and the
harness already has every seam but one. What exists: a per-request fact
channel from the engine to the seal (`Levers.turn_extras` →
`Turn.turn_extras`, I6) and from the seal to the replay row
(`TokenBatch.doc_turn_extras` → `ReplayRows.facts`, threaded adapter-blind
by the learner, `torch_learner.py:349`) — plora's recorded latent rides it
today. What does not exist: a way for the CALLER to hand the engine a
per-request choice. `PoolClient.sample(messages, stop)` and
`Engine.sample_tokens(messages, sampling, stop, bundle_id, seed)` carry
messages and spec-level knobs only. So the one addition is a typed
`directives` keyword on `sample` and `score`, carried on `Request`, and the
window's whole life is then: the environment passes `SteerWindow(start,
end)` in real-token coordinates → `apply` resolves it against the request
(offset by `Request.occupied`, the positions other adapter types put in
front — a soft prompt's rows are in the batch slice too) and writes the
resolved window into `extra_args` for the hook and into `turn_extras` for
the seal → the replay wrapper masks its add to those row positions off
`facts`. Two rules follow, both from precedent. (i) A trajectory's turns
must AGREE on the window — one forward gives each position one hidden
state, so a context steered one way for turn k and another for turn k−1 is
not replayable — and disagreeing turns are refused with the row named,
which is plora's "one trajectory is one draw" (`plora_torch.py:344`). (ii)
The prefix-cache salt carries the window beside the bundle: a block
prefilled under one window is wrong for another exactly as it is wrong for
another bundle. Score traffic takes the default (every position) unless the
scorer passes a directive — the hinted and teacher processors can. The
directive is TYPED (a frozen record the adapter type declares), never a
mapping: the wire encodes it by adapter type name and the adapter type
decodes its own. Three things this deliberately does not add: a per-token
column (the window is an interval, one fact per turn covers it), a
`segments` tensor on the row plan (an interval in absolute coordinates needs
no turn map), and a learner change (facts already flow).

**Q2a. Where the window enters.** The refold above makes the caller pass it
per request; the alternative is a window in the spec (`steer(...,
window=(start, end))`, static for the run, no keyword anywhere).
Recommendation: **the per-request directive, as refolded — `sample(...,
directives=(SteerWindow(...),))` — with "every position" as the default when
none is passed.** It is what "at runtime" means, and the blast radius is one
keyword on two client verbs, two engine verbs and two wire frames, all
defaulting to empty; every existing call site is unchanged. The spec form is
a one-line addition later if wanted (a default directive declared in `init`),
but a static window would not be runtime, which is the ask.
If the other branch: the window lives in the spec, hashes into run identity,
and no client verb changes — and changing it means a new experiment.

> **Samarth:** agree — "yea Q2a makes sense. just make sure that its possible
> to do stuff like steer constnatly on decode, etc. (this should interface
> nicely). by steer constnatly on decode, i mean whatever vllm lens did (i
> think, or the other one). go ahead and implement"

**Folded.** Steering constantly on decode is vllm-lens's behavior (every
position, prefill and decode; IBM's is last-token only) and it is this
design's DEFAULT: no directive means every position of the request, each
generated token included, at every decode step. A `SteerWindow(start=
len(prompt))` is the completion-only form of the same thing, and any window
with `end=None` runs to the end of generation. The hook applies at every
forward the request appears in — prefill and each decode step — inside the
recorded window, so "constantly on decode" needs no second mode.

**Q3. The prefix cache.** A steered prompt's KV under bundle A is wrong for
bundle B, and vLLM's block hash does not know a hook ran (LoRA and prompt
embeds are salted natively — the LoRA id and the embeds digest are extra
hash keys; #46 verified the latter). IBM's worker never mentions it, which on
a multi-tenant engine with the V1 default caching ON is a silent cross-bundle
alias. vllm-lens sets `skip_reading_prefix_cache` on every steered request.
Recommendation: **`cache_salt = bundle_id` on every request pinned to a
bundle that carries a steer entry, contributed by the lowering and merged by
the bus into whatever prompt form the request has.** vLLM injects the salt
into the first block's hash and every later block chains on it, so reuse
survives WITHIN a bundle — a GRPO group's eight completions share one
prefill — and dies across bundles. This is `EnginePlugin.cache_salt`'s rule
(`plugin.py:83`) executed through the request rather than the plugin. It is
also why `Levers` grows the field rather than the prompt-claiming lowering
owning it: `soft_prompt` claims the prompt form, and the salt is not a form.
Whether the embeds prompt form accepts `cache_salt` is verified in the pinned
image at implementation and stated in CONTEXT.
If the other branch: skip reads (correct, and every group re-prefills its
prompt eight times), or disable caching on any build serving steer (correct,
and the lora tenant beside it pays too).

> **Samarth:** agree — "i agree, we should salt the cache with the bundle id
> obfviously." (Folded with Q2: the salt is the bundle id PLUS the resolved
> window, `f"{bundle}/{start}:{end}"`, for the same reason.)

**Q4. How does a bundle's vector reach the worker?** Three shapes exist:
vllm-lens pushes per REQUEST by `collective_rpc` before generate; IBM names a
FILE in `extra_args` that the worker loads on first sight and caches; the
plugin contract imagined a per-slot bank pushed at load.
Recommendation: **the file, named per request, loaded on first sight into a
bounded per-worker bank keyed by path — the `LoRARequest` precedent exactly
(a path the worker loads and caches under `max_loras`).** `attach` is
synchronous and has no engine handle, and `add_bundle` is called from the
trainer inside the running loop (`daemons/trainer.py:177`), so a push by
`collective_rpc` (async on the V1 client) would need a new build fact and an
async arm in the bus; the file needs neither. The workdir is a tempdir in the
engine's container and every worker is a subprocess on that box (tensor
parallel included). The bank's bound is `max_bundles`, the engine's own
residency bound; a path removed by `detach` is evicted first; a path
referenced by a request in the current batch is never evicted. A bundle's
whole steer state is `n_layers × d` numbers (0.6B: 28 × 1024 × 2 bytes,
57 KB), so first-sight loading is a disk read nobody measures.
If the other branch: the push — `ServingBuild` gains an `rpc` callable, the
bus grows `await settle()` before generate, and the worker holds a slot bank
with no file — cleaner on paper, one more seam to keep, and a second way for
the engine and worker to disagree about what is resident.

> **Samarth:** agree — "i agree with your suggestion."

**Q5. Which seam claims the worker: `worker_cls` or `worker_extension_cls`?**
vLLM's extension mechanism composes by ADDING methods to the worker and (to
be confirmed against the pinned image at implementation) refuses an
extension that redefines a worker attribute, so an extension cannot override
`load_model`; the hook must then be armed by an RPC from the client — what
vllm-lens does on the first steered request.
Recommendation: **`worker_cls`: a subclass of the V1 GPU worker whose
`load_model` calls the base, PROBES, and installs the hooks.** I7 wants the
refusal at boot, in the process that would run the forward, with the seam
named — and one plugin per build is already the contract (`Seam.claim`
refuses a double claim). The class is named by string in `demands()`; rlstack
never imports it.
If the other branch: the extension plus `await collective_rpc("install")`
from an async `_ensure_llm` — composable with a second extension nobody has,
and a probe that runs at first sample instead of at boot.

> **Samarth:** delegated — "decide what you think is good". Resolved by the
> agent: **`worker_cls`**, as recommended — the probe belongs at boot in the
> process that runs the forward (I7). One obligation stated: the first
> implementation commit confirms in the pinned image that vLLM's V1
> multiprocess executor resolves a custom `worker_cls` by string in the
> worker process; if it does not, the fallback is the extension plus the
> install RPC, and CONTEXT says which landed.

**Q6. Eager mode and the model runner are DEMANDS.** Hooks do not fire inside
a captured CUDA graph; IBM's guard skips the add during capture, which means
a replayed graph steers nothing, silently. The shim reads the V1 runner's
request table; vllm-lens raises on the V2 runner.
Recommendation: **`demands()` names `enforce_eager=True`, and the probe
refuses any runner without the seams it reads (the V2 runner among them) —
so the payment is STATED, not assumed from today's default.** The fleet
already runs eager (`vllm_engine.py:52`, `residents.py:84`), so the cost is
zero today; the demand is what makes a future graph-captured build refuse
`steer` at construction instead of serving it wrong. Consequence stated: a
host that serves `steer` forfeits graphs for every tenant on it, lora
included — a host-level cost of a tenant-level choice, which `serves` already
is (a build pays for what it serves, #48 delta 3).
If the other branch: rely on the default and the guard — and the day
`enforce_eager` flips for throughput, every steer tenant on the fleet trains
against a rollout that never steered.

> **Samarth:** agree — "sure"

**Q7. The bank rule, enforced.** Two entries of one adapter type at one site
in one bank are summed today, silently, for every adapter type
(verified above). For an additive vector the sum is well defined; it is also
"two policies wearing one version number" (`soft_prompt_torch.py:231`, which
refuses its own multi-site case for that reason).
Recommendation: **a Phase-0 check, `site-overlap`, for ALL adapter types: no
two bank entries may resolve to a common site.** It is the rule
`replay.py:117` already states; it changes which specs are ACCEPTED, never
identity; the fix for a user is to make one entry (a wider pattern, or
`tie`). Across tenants nothing changes — that is I8, and the wrapper's roster.
If the other branch: the sum is declared the semantics, ARCHITECTURE.md's
bank sentence is rewritten to say so, and LoRA inherits a composition nobody
designed.

> **Samarth:** agree — "yea wait this should already be the case." (It was
> the documented rule and never a gate. Folded: `site-overlap` is the FIRST
> implementation commit, on its own, before any steer code — a gate bug fixed
> for every adapter type, with the overlapping-lora case pinned as its test.)

**Q8. New boundary sites now?** After attention (`model.layers.<n>.self_attn`
output) and the layer input are residual points both sides expose; only
`resid_pre.<n>` and `final_hidden` are in the schema.
Recommendation: **not in this ADR.** The schema fingerprint is in every run
manifest (`loop.py:155`: "same base name, different schema → loud"), so
widening the catalog refuses attach to every existing run; the inventory test
pins the count (`tests/test_siteschema.py:21`). Both lowerings address a
boundary by path, so a later commit that adds `resid_mid.<n>` to the
compilers and to `reaches` is the whole change — a schema change, its own
commit, with the manifest consequence stated.
If the other branch: add them here; every run on the volume is un-attachable
under the new schema unless the compiler is versioned, which is a bigger ADR
than this one.

> **Samarth:** agree — "yea let's do that in a later ADR. quick question: with
> this new self_attn output, will that remove the need for the custom flash
> attention rewrite for the logit bias?"

**Answer: no.** The attention-logit bias (`attn_bias`, the `queries ->
prompt[:n]` rectangle) changes the scores BEFORE the softmax, so it re-weights
which keys each query mixes — a per-query, per-head, data-dependent change to
the attention pattern whose effect on the output is `Σ_j (softmax(s+b)_j −
softmax(s)_j) · v_j`. A vector added at the attention module's OUTPUT is a
constant shift per position that does not depend on the keys or the query at
all; no choice of it reproduces a re-weighting, even for one query, let alone
for every query in a batch. So a `resid_mid.<n>` site buys "steer the stream
after attention", which is a steer, and not "bias the attention", which still
needs the kernel seam — the score_mod route or the LSE shim the #46 CORRECTION
measured. What the residual lever does change is the COST of that judgment:
side attention was refused because a plugin meant a kernel; a plugin now means
whatever seam the mechanism actually needs, and for attention bias that seam
is still inside the kernel.

**Q9. The plugin contract's per-forward verb.** `EnginePlugin.attend(view, q,
out, lse)` is abstract and attention-shaped; a residual plugin adds, it does
not attend.
Recommendation: **`EnginePlugin` keeps the LIFECYCLE abstract (probe,
install, load, evict, cache_salt) and the per-forward verb becomes the
mechanism's own — `attend` moves to `SideAttention`, `add(view, layer_idx,
hidden)` is the steer's — with the shared law kept in the docstring: written
against `BatchView` only, never engine internals.** The fakes tests move with
the verbs.
If the other branch: `SteerWorker` implements `attend` by raising, and the
contract says one thing while its only real instance does another.

> **Samarth:** agree — "yea this makes sense to m"

**Q10. The check.** `deploy/steer_l4.py`, modeled on `plora_l4.py`'s specs
and the recovered `gsm_a100.py`'s desk wiring: a CPU `Desk` container with
its OWN fleet journal; one L4 metal container (`MetalService`, `Builds(engine
= EngineBuild(serves=("lora", "steer"), enforce_eager=True))`) that announces
itself at bring-up and whose keepalive input AWAITS `until_released()`
rather than sleeping forever (ADR 0003 Q3); entrypoints `up` (spawn the
keepalive = the knock; wait for registration), `probe` (Promises 1–2, one
container, no desk), `run` (Promise 3: two specs through `RemoteDesk.submit`
— Qwen3-0.6B, the DAPO task set already in the CAS, GRPO, two updates; bank A
`{"pi": lora("layers.0-27.self_attn.*", r=16)}`, bank B `{"nudge":
steer("resid_pre.8-20", d=1024, init_std=0.02)}` with its own LR through
`OptimSpec.overrides`, the soft prompt's lesson), `down` (Promise 4:
`RemoteDesk.release`, then `status()`, then the spawned call observed to
return), `status`.
Recommendation: **run all four in that order, and add one last step: after
`down`, `desk.resolve(demands)` for the same demands, which KNOCKS the
released metal (ADR 0003 Q4) — the container boots, announces, and
`status()` shows it carve-able again — then `release` once more to end.** It
is the other half of ADR 0003's unproven list and costs one more container
boot; it is promised only if it runs green, and stays UNPROVEN in CONTEXT if
it does not.
If the other branch: stop at the release; the knock back stays unseen on the
venue, as ADR 0003 left it.

> **Samarth:** agree — "sure. id like to ideally test this under different
> conditions (TP, etc.) shortly. just make sure at the end of tests, all
> metals are deallocated."

**Folded, two ways.** (i) Promise 5: every entrypoint that acquires metal
releases every metal through the desk in a `finally` and then ASSERTS from
`status()` that nothing is on the plane — a test that ends with metal
standing has failed, whatever else it showed — and a `sweep` entrypoint
releases whatever a dead run left. (ii) The deploy takes `--tp` and `--gpu`
(the `tp_l4` precedent, #45), so the TP condition Samarth wants shortly is
one flag on the same script and a second L4; it stays a non-promise of THIS
ADR and lands as a CONTEXT line when run.

**Q11. Races, the crash midway, and the unknown slot.** (a) `detach` removes a
file a request might still name; (b) `attach` can die with a half-written
file; (c) a worker can be asked for a path it does not hold (a request in
flight across a restart, a file removed early).
Recommendation: **(a) is already impossible — `pinned()` spans the whole
generate loop (`residency.py:75`, `vllm_engine.py:233`) and residency never
evicts a pinned bundle, the LoRA guarantee; (b) `attach` writes to a
temporary name and renames, and `detach` tolerates a missing dir, so a
crashed attach leaves nothing a restore would mistake for the bundle
(content-addressed ids are the proof either way, `compile.py:73`); (c) the
hook RAISES on a named path the bank cannot load — the row plan's rule
(`replay.py:107`: an unrouted forward is a wiring bug, never a fallback) on
the engine side — and a request with no `rlstack_steer` key adds nothing,
which is how a lora-only tenant rides the same batch.** Resume: no new
records (the vector is deterministic per position, so nothing is drawn), the
emitted bytes are a pure function of the parameters, and `test_resume.py`
gains the steer case; the ledger, blobs and manifest carry nothing they did
not.
If the other branch: a missing path skips silently and a tenant trains
against rollouts that steered some requests and not others, undetectably.

> **Samarth:** delegated — "just decide what's best for these races."
> Resolved by the agent: **as recommended, all three** — (a) `pinned()`
> spans the request, so detach under a live request cannot happen; (b)
> write-then-rename on attach, a tolerant detach, content-addressed ids as
> the proof; (c) an unknown path RAISES in the hook, a request without the
> key adds nothing. One addition from Q2: a window that does not fit the
> request (start beyond its length, end before start) is refused at `apply`
> with the request named, never clamped — a clamped window would record
> something the caller did not ask for.

## Outcome

**Landed** (CONTEXT #77), 2026-09-02/03, in the commit order the ADR set:
the `site-overlap` gate first and alone; the directive and the two levers;
the adapter type on `Mechanism.RESIDUAL`; the engine-image half
(`SteerPlugin`, `SteerWorker`, `BatchView.from_vllm` built, the contract's
per-forward verb per mechanism); the deploy; then what metal taught.

- **The answers changed the shape twice.** Q2 (Samarth: positions at
  runtime) became the `Directive` — typed, per adapter type, carried on the
  sample and score verbs and the wire, recorded at the seal by
  `record_directive` — with `SteerWindow` as the steer's and "every position,
  every decode step" as the default (Q2a). Q6's demand grew a third form on
  metal: `BuildDemands.env`, because vllm 0.28.0 chooses the model runner by
  environment variable alone and boots V2 by default; `check_demand_fits`
  is the pure rule that refuses a demand contradicting a build fact.
- **What metal found that the fakes could not.** A steer wrapping a decoder
  layer hid the layer's children from the walk to a lora inside it
  (`leaf_module` walks through a wrapper on the way); the V2 runner; the
  worker's boot step must return the warm-up's reply. On the venue: a host
  reaching a pool on its own metal by a Modal self-call wedges the container
  (same-metal addresses are in-process now), a killed driver kills the metal
  (a cancellation propagates), drivers must print unbuffered, and a released
  container is a zombie until the venue's scaledown (the shift now ends with
  `stop_fetching_inputs`, and a door landing on a released container stands
  a fresh metal up).
- **Promises, checked.** 1: the L4 probe's zero-tolerance control passed on
  both sides (max |Δ| 0.00e+00). 2: gaps 0.042–0.059 at three magnitudes,
  0.052 beside a lora (the lora alone: 0.067), shift control 20–30× the gap;
  the window replays as recorded; the cache never aliased; decode is steered
  under an open window and not under a closed one. 3: two tenants through
  `RemoteDesk.submit`, the second joined the first's listings, two updates
  each, rails 0.017–0.023 (loss 0.0: all-or-nothing groups on an unscreened
  draw), then again on the plora venue's screened problem with a non-zero
  loss on both tenants (lora 0.0001 / −0.0001, steer 0.0002 / 0.0003; rails
  0.013–0.021). 4: `RemoteDesk.release` told, the keepalive returned 0.1 s
  later after a 248 s shift — the first observed release on Modal — and
  again after 429 s. 5: the plane asserted empty, every time, a failed run's
  `finally` included. The window round trip, the resume case
  with a steer bank, the plugin lifecycle and the demand refusals are pinned
  in the fakes. Tests: 889 locally from 835 (+54; 111 torch-gated skips),
  885 green in the image.
- **Unproven, as the non-promises said:** TP above 1 (one flag away),
  pipeline parallel, the knock back after release (`::knock` written, not
  run), the throughput cost of eager plus hooks, mid-layer sites (Q8's later
  ADR).
