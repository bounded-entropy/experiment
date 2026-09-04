# ADR 0007 — One desk, bare metal, transports in rlstack: the venue chassis, with `concept_steer.py` as its proof

| | |
|---|---|
| **Date** | 2026-09-04 |
| **Status** | Implemented on fakes (CONTEXT #84); awaiting the metal re-runs (Q9). Accepted 2026-09-04: Q2, Q3, Q4, Q4a, Q8, Q9 agreed; Q6 agreed with a rider, folded; Q1, Q5, Q7, Q10, Q11 stood on their recommendations |
| **Author** | Claude Fable 5.1 (session: the SPAR introspection paper, 2026-09-04) |
| **Touches** | `rlstack/runner/transports/` (NEW region: one file per wire substrate, `modal_cls.py` first), `rlstack/runner/remote.py` (a transport factory keyed by address scheme; `LocalTransport` stays), `rlstack/runner/desk.py` (`MetalService` boots BARE; the recipe is a desk-held, journaled row; `knock` refuses loudly without `boot_for`), `rlstack/data/stores/` (the observer's read-only Modal view folded in or retired), `deploy/desk.py` (NEW: the one standing desk and its doors), `deploy/modal_venue.py` (NEW: the metal container chassis and the campaign helpers), `deploy/concept_steer.py` (rewritten minimal — the proof of concept), `deploy/steer_l4.py` + `deploy/stress_fleet.py` (rewritten onto the chassis; their probes are the regression proof), `deploy/dapo_grpo.py` + `deploy/plora_l4.py` (retired, Q8), `deploy/ui.py`, `STYLE.md` rule 8 (one line: the new region; Samarth's edit), `ARCHITECTURE.md` (Wire, Desk, Builds, MetalService entries), `tests/test_architecture.py` (the region, and rule 7 pinned), `tests/` |
| **Invariants** | I5 (topology is semantics-neutral: a venue owns addresses and images, never a recipe or a builder), I7 (the substrate is certified: a transport is a build-side implementation behind a protocol), I12 (a metal's capability is a birth fact — measured; its RECIPE becomes the desk's declaration, not the container's), I10 (the fleet journal is one record for one fleet) |
| **CONTEXT** | extends #43 (the fleet), #68 ("concurrent campaigns serialize through one desk" — stated, never built), #69 (the blind desk), #74 (residents), #76 / ADR 0003 (release, the knock), #77 (the venue's transports and the wedge), #79 (the routable learner); the 2026-09-04 portability audit (scratchpad report, folded below) |

## Original prompt

> "Transports" -> we should move this out into smth more general: a transport interface that has modal transports implemented as a specific implementation, right?
>
> "The Desk container" -> we should have a general desk interface already and have modal be a specific implementation, and have its own separate start script. how would multiple deploy scripts even work right now, is it like, the other deploy scripts have to explicitly declare the desk container or smth?
>
> why do we have a bring up metal here? that should entirely belong to the desk
>
> what about other things? i think in general, we need a lot of cleanup and abstrcation here (but we're pretty close probably, we're just implementing things badly)

> "deploy/modal_transport" why should it live in deploy? should probably be in the main codebase in rlstack (and the main codebase can have other implementations for other transports, like when we eventually have AWS). by the way, is it generally true that all the functionalities of our codebase that we're dependent on can be easily implemented for other frameworks, like AWS or runpod or smth? i just want to get a scope of where things are at.

> yes, ADR 0007 should be the cleanup that does all of this. as proof of concept, it should consolidate the concept_steer deploy to be minimal. could you create the ADR?

## Context / problem

**Every desk-shaped venue fuses three services into one file.** `deploy/concept_steer.py` is 857 lines of which the experiment — two topologies, two plans, two specs, one measurement — is about 110. The rest is a Modal desk container (55 lines), a Modal metal container with its bring-up, announce, duties, keepalive and rebirth (145), three Modal `Transport` implementations (65), door helpers (80), the doors' spawn-wait-take-down ceremony (110), images and constants (90), and a docstring (55). `deploy/steer_l4.py` (948) and `deploy/stress_fleet.py` (976) are the same file with different constants: the portability audit diffed the transport blocks and found `concept_steer`'s byte-identical to `steer_l4`'s and `stress_fleet`'s identical minus three docstrings (F1). A fourth, older copy lives in `deploy/dapo_grpo.py:167-178`, and `ARCHITECTURE.md:629` still cites their home as `deploy/modal_host.py`, a file that no longer exists.

**Each venue is its own fleet.** Each of the three files stands up its OWN Modal app, its OWN `Desk` container and its OWN fleet journal (`fleet/steer.jsonl`, `fleet/stress.jsonl`, `fleet/concept.jsonl` — `steer_l4.py:102`, `stress_fleet.py:92`, `concept_steer.py:148`), and its metal registers only with that app's desk. So the answer to "how do multiple deploy scripts work" is: they do not share anything. A campaign can only place onto metal that registered with ITS app, three fleets share one volume without seeing each other, and #68's own design statement — "concurrent campaigns serialize through one desk instead of double-reading one residual" — was recorded and never built. The per-journal isolation was deliberate when the gsm and dsl venues shared a volume (`gsm_a100.py`'s docstring: "two desks must not replay each other's listings"); it solved a collision by forbidding the sharing the desk exists for.

**The recipe is declared by the wrong party.** `bring_up_metal` (`concept_steer.py:288-327`) constructs `MetalService(measure(METAL), store=…, builds=Builds(engine=EngineBuild(max_model_len=4096, serves=("steer",), enforce_eager=True, enable_sleep_mode=True), learner=LearnerBuild(…)), address_of=…, schema_for=hf_schema, transport_for=…)`. Of those arguments, measuring the card, the store mount and the keepalive must run on the metal; `schema_for` is universal (`hf_schema`); `address_of` and `transport_for` are venue plumbing; and `builds` — the recipe — is the one that should not be there. The code already half-agrees: the desk's journaled recipe row is canon (ADR 0001 Q5c), it rides every carve request, and `MetalService.adopt_recipe` (`desk.py:1507`) replaces whatever the metal booted with. The only reason the venue declares one is that `MetalService.__init__` (`desk.py:1340`) requires `builds` at construction.

**Four more things the audit found on the same seam.**
- F3: `Desk.knock` (`desk.py:974-998`) boots a released or dead metal through `boot_for` when the venue supplies one, else through `describe()` on the plane address — "on Modal that call IS the boot". No venue supplies `boot_for` (`git grep boot_for -- deploy` is empty), and on any venue whose containers do not boot lazily, the reap → knock → re-register → reroute loop degrades to "the metal stays down" as a silent no-op.
- F4: `deploy/ui.py:41-107` holds a complete second Modal `Store` backend (`VolumeReadStore`, all seven byte verbs over the Volume SDK plus a listing snapshot) outside `data/stores/`, outside `open_store`, outside the fakes suite; `steer_l4.py:184` subclasses `ModalVolumeStore` to re-key the fleet journal for the same per-venue-desk reason.
- F5: STYLE rule 7's lazy-import discipline (heavy substrates imported at module scope only inside their region) holds by hand — no test pins it. A third heavy region, `runner/transports/`, is where one stray module-scope import would make `import rlstack` require `modal` and kill the stdlib-only fakes suite.
- The doors' take-down ceremony (`take_down`: release every metal, wait for the keepalive to return, assert the plane empty) was built for the CHECK venues, where "the plane ends empty" is a promise (ADR 0003 promise 5, #77). Under one standing desk a CAMPAIGN door must not release the fleet; ADR 0003's idle rule already does that. `concept_steer.py::distill_set` and `::train` copy the ceremony anyway (`concept_steer.py:768,787`).

**What the audit found to be already right, so this ADR does not touch it.** `Transport` is two verbs with JSON-safe frames by contract, and `LocalTransport` round-trips every frame through `json.dumps` both ways, so the fakes suite is a wire test (#45 measured a real transport at "four lines of body"). `build_engine` / `build_learner` are universal (ADR 0002 Q4); a venue writes no builder. Measurement, carve/decarve/release, placement, the journal, resume, retention and the observer are venue-blind; `desk.py` names Modal in two docstrings. A POSIX-mount venue touches two lines inside `rlstack/` (`address.py:open_store`, `observe/locate.py:store_for`). S3-as-a-store and multi-node FSDP are real, later ADRs (the append-only ledger; the localhost rendezvous), not this one.

**Unmeasured.** Nothing here is a number; every claim is a line count or a diff.

## Decision

**Three services, one chassis, transports in the library.** The desk becomes ONE standing service with its own start script and one journal; the metal boots BARE and takes its recipe from the desk; the Modal transports move into `rlstack/runner/transports/` as one implementation of the existing protocol; what remains venue — images, apps, addresses, the metal container's Modal wrapping, the doors' helpers — lives once in a chassis; and a venue file is its constants, its specs and its doors. Concretely:

- **Transports (Q1).** `rlstack/runner/transports/modal_cls.py`: ONE class, `ModalClsTransport(app, cls, address=None)`, implementing `Transport.call` / `ask` over `modal.Cls.from_name(app, cls)`'s `call`/`ask` methods (the three copies today differ only in which class and method they hit and whether they carry an address). `import modal` at module scope, the region imported lazily from the package root — `runner/engines/` and `runner/learners/`' rule (STYLE rules 7 and 8). `LocalTransport` STAYS in `remote.py` beside the protocol: it is the contract's enforcement (`json_roundtrip`), not a substrate, exactly as `FakeEngine` lives in `runner/fakes.py` and not in `runner/engines/` (the audit's §5, agreed). `remote.py` gains `transport_for(address) -> Transport`, a factory keyed by the address's scheme (Q3), so the desk's `host_for` / `metal_for` resolvers and a host's `transport_for` are the same function everywhere.
- **The address grammar (Q3).** Addresses carry the venue: `modal://<app>/<cls>#<host>` for a host's door, `modal://<app>/<cls>` for a metal's plane, `local://<host>` for the in-process wire. A desk that serves many metal apps needs the app IN the address; today's `<scheme>://<host>` with the app baked into each venue's transport class is why one desk could not serve two venues.
- **One desk (Q2).** `deploy/desk.py`: the Modal app `rlstack-desk`, one `Desk` container rebuilt from ONE journal (`fleet/log.jsonl`, the default the store already has), `host_for` / `metal_for` = `transport_for`, `boot_for` = the Modal knock (Q5). Doors: `status`, `sweep` (release every metal; the operator's backstop), `recipe` (Q4), `release --metal`. Deployed once: `modal deploy deploy/desk.py`. Every venue's metal registers with THIS desk by name, and every campaign submits to it: `RemoteDesk(transport_for("modal://rlstack-desk/Desk"))`. The per-venue `Desk` classes and journals are deleted; old journals stay on the volume as history and are never read by the new desk.
- **Bare metal, the desk's recipe (Q4).** `MetalService.__init__`'s `builds` becomes optional: a metal with no recipe has an empty carve-able set until a carve request delivers one (`adopt_recipe` already exists). The desk holds the recipe as a journaled `recipe` event per metal name — set by the `deploy/desk.py::recipe` door (the operator's declaration: `--metal concept-a100 --engine max_model_len=4096,serves=steer,enforce_eager,enable_sleep_mode --learner checkpoint_activations`) or carried by `register_metal` when a metal file chooses to declare one — and the carve carries it, as today. ARCHITECTURE's "declared at bring-up from the deploy's constants — the metal's FIRST declaration" becomes "declared at the desk; a metal's constants are at most a proposal the desk journals". Q4a asks whether the recipe's adapter-type half should be DERIVED from demands instead.
- **The metal chassis.** `deploy/modal_venue.py`: `metal_class(app, name, gpus, scheme)` returning the Modal class that `concept_steer.py`'s `MetalS` is today — bring-up (measure, store, the same-metal `LocalTransport` rule from #77), announce (register with the ONE desk, carrying `idle_s` and, optionally, a recipe), duties (stats, volume commit), the keepalive `serve` as the shift (ADR 0003 Q3), rebirth on a released container (ADR 0003 Q4), teardown. A venue's metal file is `MetalS = metal_class(APP, "concept-a100", ["A100-80GB:2", "H100:2"], …)` plus images. Also in the chassis: the campaign helpers — `submit_and_follow(spec, subdir)` (submit through the desk; follow `run_progress` to the extent), `progress`, `export_blob`, `wait_for_metal`.
- **The knock (Q5).** The Modal chassis supplies `boot_for(name)` — `spawn()` the metal's keepalive, which is what `::up` does by hand today — so the desk's knock is explicit on every venue, and `Desk.knock` REFUSES loudly (a journaled `knock-refused` event and a returned reason) when neither `boot_for` nor a plane address exists, instead of returning `False` silently.
- **Doors under a standing desk (Q6).** A campaign door submits and follows; it never releases. Idle metal is the desk's to release (ADR 0003). The CHECK venues (`steer_l4`, `stress_fleet`) keep an explicit `release` at the end of each door because "this metal released, the plane empty of MY metal" is their promise — but they assert emptiness of the metals they registered, not of the whole plane, which under one desk may hold other people's metal. **And an explicit release from a door is GUARDED (Q6, Samarth's rider):** like `decommission`, it is refused — journaled and returned with the running work NAMED — when any tenancy that is not the door's own is running on, or routing through, that metal; `force` stays an operator's verb at `deploy/desk.py` alone. A door tearing down metal it acquired never takes another experiment with it.
- **The observer's store (Q7).** `deploy/ui.py`'s `VolumeReadStore` is retired: the UI container mounts the volume and runs `python -m rlstack ui /store` over `LocalStore` (the audit's finding that on a POSIX mount the class collapses to that). If a read-only Modal view is still wanted, it lives in `data/stores/modal_volume.py` behind `open_store`, covered by the fakes — never in `deploy/`.
- **The hand-built venues (Q8).** `deploy/dapo_grpo.py` and `deploy/plora_l4.py` bypass the desk and construct hosts by hand in the driver's container. They are retired (deleted; history keeps them, their CONTEXT entries stand); the desk-shaped venues cover both shapes (#77 the plora problem, #78 tp+fsdp).
- **The proof (Q9).** `deploy/concept_steer.py` rewritten to its constants, its recipe proposal, its images, `MetalS = metal_class(…)`, its four spec/plan/measurement functions, and its doors (`prompts`, `distill_set`, `train`, `measure`, `export`) — target ≤ 220 lines, every line either science or a door. `steer_l4.py` and `stress_fleet.py` rewritten the same way, and their metal doors (`steer_l4::probe`, `::check`; `stress_fleet::topology`, `::latejoin`, `::learner_sleep`) are the regression proof Samarth runs.
- **Rule 7 pinned (Q10).** `tests/test_architecture.py` gains the check the audit sketched: no module under `rlstack/` outside `runner/engines/`, `runner/learners/`, `runner/transports/` (and `rlstack_engine/`) imports those regions or their substrates (`torch`, `vllm`, `modal`) at module scope; `import rlstack` in a stdlib-only interpreter stays green — which the local suite already proves every run.

### Touched / untouched

- **Touched** — `rlstack/runner/transports/__init__.py` (the region's charter, mirroring `engines/__init__.py`), `transports/modal_cls.py` (the one Modal transport; `import modal` at module scope). `rlstack/runner/remote.py`: `transport_for(address)` (scheme → transport; `local://` → the in-process service registry a metal keeps; `modal://` → `modal_cls`, imported inside the branch), the address grammar in one place (`parse_address`). `rlstack/runner/desk.py`: `MetalService(builds=None)`; `Desk.recipes` (journaled `recipe` events, `recipe_for(name)`), `register_metal(builds=…)` optional, the carve carrying `recipe_for`; `knock` refusing loudly; `from_journal` reading `recipe` events. `rlstack/data/stores/base.py`: nothing, unless Q7's read-only view is kept (then `modal_volume.py`). `deploy/desk.py`, `deploy/modal_venue.py` (new). `deploy/concept_steer.py`, `deploy/steer_l4.py`, `deploy/stress_fleet.py` (rewritten). `deploy/ui.py` (the mount). `deploy/dapo_grpo.py`, `deploy/plora_l4.py` (deleted). `ARCHITECTURE.md`: the Wire entry (`deploy/modal_host.py` → `runner/transports/`), the Desk entry (one desk, `recipe`), the Builds entry (declared at the desk), the MetalService entry (boots bare), the "Campaign layer" entry (submit and follow; never release). `STYLE.md` rule 8: one line for `runner/transports/` and the stale `runner/sources/` line — **Samarth's edit**, flagged. `tests/test_architecture.py` (the region; rule 7). `tests/test_desk.py` (bare metal, the recipe row, the loud knock), `tests/test_remote.py` (the factory and the grammar), a fakes-backed chassis test (Q9).
- **Untouched** — `rlstack/runner/host.py`, `campaign.py`, `residents.py`, `loop.py`, the daemons, `data/plan.py`, `refs.py`: the workload plane does not know the fleet's shape. `Desk.place` / `submit` / `deliver` / `reroute` / `reap` / `release`: the ladder is unchanged; only the knock's refusal and the recipe's source move. `runner/engines/`, `runner/learners/`: nothing. `rlstack_engine/`: nothing. `data/stores/modal_volume.py`'s `ModalVolumeStore`: unchanged (Q7 adds or not). The per-venue journals on the volume: left as history. Every spec, plan and measurement in `concept_steer.py`: byte-identical values, so every run identity is unchanged. `tests/test_resume.py`: untouched.

### Promises / non-promises

- **Promises** — (1) `deploy/concept_steer.py` ≤ 220 lines with every remaining line science or a door; `steer_l4.py` and `stress_fleet.py` each lose their transport, desk and metal blocks (≥ 350 lines each). (2) Exactly ONE `Transport` implementation for Modal exists in the tree, under `rlstack/runner/transports/`; `git grep "class .*Transport" -- deploy` is empty. (3) `import rlstack` and the fakes suite stay stdlib-only, pinned by the new architecture test. (4) One desk app, one journal: `deploy/desk.py` is the only file defining a desk container; the three per-venue `Desk` classes and `FLEET_LOG`s are gone. (5) A metal constructed with no recipe registers, is listed, and carves correctly once the desk delivers a recipe — on fakes (`tests/test_desk.py`), with the recipe row journaled and rebuilt by `from_journal`. (6) `Desk.knock` on a metal with neither `boot_for` nor an address journals a refusal and returns it; on the Modal chassis `boot_for` spawns the keepalive (tested with a fake boot). (7) Every spec value in the rewritten venues canonicalizes to the same row as before the rewrite (a test compares `canonical_json` of `concept_steer.teacher_spec` / `student_spec` before and after, via a fixture of the pre-rewrite rows). (8) The fakes suite is green. (9) A door's release of a metal that carries another run's tenancy is refused with that run named, on fakes; the same release with only the door's own tenancy finished proceeds.
- **Non-promises** — No metal is run by this ADR; the check venues' doors are the regression proof and Samarth runs them. Nothing here makes a second venue exist: `transport_for` has one real scheme, and `open_store` still has two backends. S3-as-a-store and multi-node FSDP are not touched (their own ADRs). The recipe's derivation from demands (Q4a) is asked, not built. Old per-venue journals are not migrated: a run placed by an old desk is resubmitted to the new one, which is a resume by identity.

### Interfaces

- **`Transport`** (`remote.py`): unchanged. **`transport_for(address)`**: the one factory; a venue never constructs a transport by class again. **Addresses**: `modal://<app>/<cls>[#<host>]`, `local://<host>`; the grammar lives in `remote.py` beside the frames.
- **The desk's doors**: `register_metal(name, gpu, devices, vram_gb, address, idle_s, builds=None)`; `recipe(name, builds)` (new, journaled); `release(name, reason, force=False)` — GUARDED by dependents like `decommission` (Q6); `status`, `liveness` as today. **`covers`** consults the listing's recipe `serves` against the spec's adapter types (Q4a). **`Desk.knock`**: `boot_for` or a loud refusal.
- **`MetalService(metal, store, address_of, schema_for, transport_for, builds=None)`**: bare until the first carve or a registration reply carrying the desk's recipe.
- **The chassis** (`deploy/modal_venue.py`): `metal_class(app, name, gpus, scheme, images)`, `desk()`, `submit_and_follow(spec, subdir, timeout_s)`, `wait_for_metal(name)`, `export_blob(run_id, section, name, version)`. Nothing semantics-bearing: it constructs specs it is handed and never builds one.
- **`observe/`**: nothing new to render; the fleet view reads ONE journal.
- **The gate**: nothing.

### Sketches

```python
# rlstack/runner/transports/modal_cls.py — the one Modal transport
import modal                                     # module scope: this region loads lazily

class ModalClsTransport:
    """Transport over one Modal class's `call`/`ask` methods. An address
    names the app and the class — and the host, when the class multiplexes
    hosts (a metal container's `host`/`host_ask` doors take one)."""
    def __init__(self, app: str, cls: str, host: str | None = None) -> None: ...
    async def call(self, verb: str, payload: dict) -> dict: ...
    def ask(self, verb: str, payload: dict) -> dict: ...      # blocking, own thread (#77)

# rlstack/runner/remote.py — the factory and the grammar, beside the protocol
def transport_for(address: str) -> Transport:
    """modal://<app>/<cls>[#<host>] -> ModalClsTransport (imported here, lazily);
    local://<host> -> the in-process service the metal registered under that
    name. An unknown scheme is refused by name."""

# rlstack/runner/desk.py — the recipe is the desk's
class Desk:
    def recipe(self, metal: str, builds: Builds) -> None:
        """Journal `recipe` for a metal; the next carve to it carries this row."""
    def recipe_for(self, metal: str) -> Builds | None: ...

class MetalService:
    def __init__(self, metal, *, store, address_of, schema_for=None,
                 transport_for=None, builds: Builds | None = None, spawn=...): ...
    # carve(request): if request carries a recipe, adopt it; a bare metal with
    # no recipe refuses the carve by name ("no recipe: declare one at the desk")

# deploy/desk.py — the one desk, its own start script
app = modal.App("rlstack-desk")
@app.cls(image=cpu_image, volumes={"/store": store_volume}, min_containers=1, max_containers=1)
class Desk:
    @modal.enter()
    def bring_up(self):
        self.desk = Desk.from_journal(a_store(), host_for=..., metal_for=...,
                                      boot_for=boot_by_spawn, idle_s=IDLE_S)   # transport_for underneath
        self.door = Campaigns(self.desk)

# deploy/modal_venue.py — the chassis
def metal_class(app_name: str, metal: str, gpus: list[str], images) -> type: ...
def submit_and_follow(spec, subdir: str, timeout_s: float) -> str: ...

# deploy/concept_steer.py — after: constants, images, MetalS, four science functions, five doors
MetalS = metal_class("rlstack-concept-steer", "concept-a100", GPUS, images)
def teacher_spec(store, train_tasks): ...      # unchanged values
def student_spec(store, teacher_run, layer): ...
@app.local_entrypoint()
def train(layer: int, teacher_run: str):
    run_id = submit_and_follow(student_spec(a_store(), teacher_run, layer), SUBDIR, 14400)
```

## Questions

**Q1. Transports move to `rlstack/runner/transports/modal_cls.py`; `LocalTransport` stays in `remote.py`.**
Recommendation: yes to both. A transport is a wire substrate, one file per substrate with its heavy import at module scope and lazy loading from the package root — `engines/` and `learners/`' rule, and the place an `aws.py` or `runpod.py` lands later. `LocalTransport` is the protocol's enforcement (`json_roundtrip`), not a substrate, and the repo keeps fakes beside their protocol. `modal_cls.py`, not `modal.py`, so the module never shadows the package it imports (the `modal_volume.py` precedent).
If the other branch (`deploy/modal_transport.py`): the venue folder keeps a protocol implementation and every new venue copies it in; rule 8's "nothing semantics-bearing lives in deploy/" is honored only nominally.

> **Samarth:** not raised in the 2026-09-04 review; the recommendation stands unless Samarth objects.

**Q2. ONE desk: the app `rlstack-desk`, one journal, every venue's metal registering with it, every campaign submitting to it; the per-venue desks and journals deleted, old journals left as history.**
Recommendation: yes — it is #68's stated design. The cost is that the CHECK venues' "plane empty" promise becomes "MY metal released": under one desk the plane may hold someone else's metal, so `steer_l4::check` asserts that the metals it registered are released, not that nothing stands.
If the other branch (keep per-venue desks, share only the chassis): the transports and metal chassis still deduplicate, but "how do multiple deploy scripts work" keeps its answer — they do not share metal — and #68 stays unbuilt.

> **Samarth:** agree — one desk (2026-09-04).

**Q3. Addresses carry the venue: `modal://<app>/<cls>[#<host>]`, and one `transport_for(address)` factory in `remote.py` replaces every per-venue `host_for` / `metal_for` / `transport_for` closure.**
Recommendation: yes. One desk serving many metal apps must find the app in the address; today it is baked into each venue's transport class, which is exactly why one desk could not serve two venues. The journal's `address` fields become self-describing; old journals' `<scheme>://<host>` addresses are unreadable by the new desk, which is the "left as history" in Q2.
If the other branch (keep `<scheme>://<host>` and a per-scheme registry the desk is configured with): the desk file grows a table of scheme → app, which is the same information in a worse place.

> **Samarth:** agree (2026-09-04).

**Q4. The recipe is the desk's: `MetalService` boots bare, `Desk.recipe(metal, builds)` is a journaled declaration set through a desk door (or carried by a registration that chooses to propose one), and a carve to a metal with no recipe is refused by name.**
Recommendation: yes. The desk's row is already canon (ADR 0001 Q5c) and the metal already adopts it on every carve; making `builds` optional at construction and adding the `recipe` event is the whole change. The refusal is the honest fallback: a bare metal with no declared recipe cannot know what to serve.
If the other branch (the metal keeps declaring at bring-up): `bring_up_metal` stays in every venue with its `serves=`, and the desk's canon is a copy of a venue constant.

> **Samarth:** agree — desk-held, declared (2026-09-04).

**Q4a. Should the recipe's adapter-type half be DERIVED from demands rather than declared?** `serves=("steer",)`, `enforce_eager`, the worker class, the V1 runner are exactly what a spec's adapter types DEMAND (`RolloutLowering.demands`, #77); only capacity knobs (`max_model_len`, `max_bundles`, `max_rank`, sleep mode) are genuinely declarations. Today `covers` (`desk.py:1274`) matches capability, base and shape and ignores `serves`, so a steer spec can be JOINED onto a listing whose engine does not serve steer and be refused late, at Phase 0 on the host.
Recommendation: not in this ADR — declare now, and make `covers` consult the listing's recipe `serves` against the spec's adapter types so the join refuses EARLY (a one-rule change with a test). Derivation is a follow-up once a second adapter type's demands conflict with a first's on one build.
If the other branch (derive now): the desk learns adapter types, which breaks its workload-blindness (#69) unless the campaign layer projects demands into the demand rows first — a bigger change than this ADR.

> **Samarth:** agree — declare now; `covers` checks `serves` so the join refuses early; derivation is a follow-up (2026-09-04).

**Q5. The Modal chassis supplies `boot_for` (spawn the keepalive), and `Desk.knock` refuses LOUDLY (journaled, returned) when it has neither `boot_for` nor an address.**
Recommendation: yes. F3: the lazy-boot fallback is a Modal semantic hiding in the desk, and on a venue whose containers do not boot on a call the supervision loop fails as a no-op. Explicit on Modal, loud everywhere else.
If the other branch: keep the fallback as the Modal default and document it; the first non-lazy venue discovers the silence in production.

> **Samarth:** not raised in the 2026-09-04 review; the recommendation stands unless Samarth objects.

**Q6. A campaign door never releases metal; the desk's idle rule does (ADR 0003). Only the check venues' doors end in an explicit release, scoped to their own metals.**
Recommendation: yes. `concept_steer`'s `distill_set` / `train` / `measure` copy the check venues' take-down ceremony, which would tear down a standing fleet after every arm; under one desk that is wrong by construction. The operator's backstop is `deploy/desk.py::sweep`.
If the other branch (every door releases what it acquired): a campaign of three arms boots the 32B three times, and two campaigns sharing a metal race each other's teardown.

> **Samarth:** "Agree, but an experiment tearing down metal it acquired should obviously not tear it down if other experiments are still running on it" (2026-09-04). *Folded: an explicit release from a door is guarded by dependents like `decommission` — refused with the running work named; `force` is the desk's own door only (Decision, promise 9).*

**Q7. `deploy/ui.py`'s `VolumeReadStore` is retired: the UI container mounts the volume and serves `python -m rlstack ui /store` over `LocalStore`.**
Recommendation: yes. F4: a second Modal store backend outside `data/stores/`, uncovered by the fakes. On a mount the class is `LocalStore` with a locator. If a read-only SDK view is ever needed again (a UI host without a mount), it goes in `data/stores/modal_volume.py` behind `open_store`, with fakes coverage.
If the other branch: keep it and file it under `data/stores/` now, which is more code for a case no venue has.

> **Samarth:** not raised in the 2026-09-04 review; the recommendation stands unless Samarth objects.

**Q8. `deploy/dapo_grpo.py` and `deploy/plora_l4.py` are retired (deleted; history and CONTEXT keep them).**
Recommendation: yes. Both bypass the desk and hand-build hosts in the driver's container — the "campaigns carrying the architecture" shape Samarth ruled against on the a69 worktree ("Clean slate: delete every deploy script"). Their science is recorded (#61, #62, #64, #77) and their shapes are covered by the desk venues. The DAPO task-set builder (`tasks_dapo.py`) stays: it is content, not a venue.
If the other branch (rewrite them onto the chassis): two more ~200-line venue files whose experiments are finished.

> **Samarth:** agree — retire them (2026-09-04).

**Q9. The proof is `concept_steer.py` at ≤ 220 lines on the chassis, a fakes-backed chassis test, and the check venues' doors re-run on metal by Samarth.**
Recommendation: yes. The chassis test: a `FakeMetal` process-free stand-in (the `FakeEngineBuild` precedent) registered with a `Desk` over `LocalTransport` through `transport_for("local://…")`, a bare metal receiving its recipe, a spec submitted and followed to its extent — all on fakes, no Modal import. The metal re-runs (`steer_l4::probe` and `::check`, `stress_fleet::topology`, `::latejoin`, `::learner_sleep`) are the regression proof that the rewrite changed no number; they are Samarth's to run and this ADR is not Implemented until at least `steer_l4::check` has passed through the ONE desk.
If the other branch (rewrite only `concept_steer`): three copies become two and the check venues keep their own desks, which is Q2's other branch by the back door.

> **Samarth:** agree — all three venues on the chassis; the check venues' doors on metal before Implemented (2026-09-04).

**Q10. Rule 7 is pinned by a test: no module-scope import of `torch`, `vllm`, `modal`, or of the heavy regions, anywhere in `rlstack/` outside those regions and `rlstack_engine/`.**
Recommendation: yes — ten lines in `test_architecture.py`'s existing AST idiom (F5). `fsdp_torch.py`'s imports of `torch_learner` and `ranks` are inside the region and stay legal.
If the other branch: the discipline holds by hand, and the first stray import in `transports/` is found by the image suite, not the local one.

> **Samarth:** not raised in the 2026-09-04 review; the recommendation stands unless Samarth objects.

**Q11. Byte-identity and the crash states, stated once.** No run identity changes: every spec value is unchanged (promise 7) and nothing here is hashed. The fleet journal moves to ONE file for NEW registrations; a desk restart rebuilds from it exactly as today (`from_journal` gains one event type, `recipe`, the latest per metal winning). A metal that boots while the desk is down registers on its next announce, as today. A carve that arrives at a bare metal before the desk has a recipe for it is refused, journaled, and retried on the next registration event like every parked placement — never a half-built host. A run placed by an old per-venue desk is not visible to the new desk; resubmitting the same spec to the new desk is a resume by identity (I3), which is the migration. Agree these are the obligations?

> **Samarth:** not raised in the 2026-09-04 review; the obligations stand as stated.

## Outcome

Landed in five commits, `4d10c40` → `6cf21c2`, and recorded as **CONTEXT #84**.
**1051 tests green, from 1003.** The fourth commit (`9ec7343`) is a correction
found in review: the first cut of `deploy/desk.py::recipe` wrote the `recipe`
event from a side `@app.function` — a SECOND writer on the fleet journal, which
the standing desk would not have seen until a restart (I10). The door now sends
the desk's `recipe` WIRE VERB (`RemoteDesk.recipe` → `Desk.serve`), and the
journal keeps its one writer.

**What landed, against the ten Decision bullets.** All ten. `runner/transports/`
with `modal_cls.py` and `remote.py`'s `Address` / `parse_address` /
`transport_for` (Q1, Q3); `deploy/desk.py` as the one desk on one journal (Q2);
`MetalService(builds=None)` with the desk's journaled `recipe` event and a
carve refused by name at both ends (Q4); `covers` split into
`matches_capability` + `recipe_serves`, fed by `campaign.adapter_types_of`
(Q4a); `deploy/modal_venue.py`'s `metal_class` and campaign helpers; the
chassis' `boot_for` and `Desk.knock`'s journaled `knock-refused` (Q5); campaign
doors that never release and check-venue releases that are guarded and scoped
(Q6); `deploy/ui.py` over the mount (Q7); the two hand-built venues deleted
(Q8); the three venues on the chassis plus a fakes-backed chassis test (Q9);
rule 7 pinned by `tests/test_architecture.py` (Q10).

**What the answers changed.** Q6's rider is the only shape change, and it grew
a second guard the ADR did not have: `Desk.release` is now refused over
dependents exactly as `decommission` is, asked of the WHOLE metal
(`metal_dependents`, over one shared `dependents_on` body) because a release
takes every host on a metal at once. Its corollary had to be ruled on
separately: `release_idle` passes `force=True`, because the idle sweep's
evidence — nothing busy on that metal for its whole limit — is stronger than
the guard's, and letting a stale "running" row veto it would have quietly
weakened ADR 0003's promise.

**Three things the implementation added that the ADR did not name.**
(1) `MetalService.route` / `unroute`: #77's in-process rule became a published
entry on `remote.py`'s switchboard rather than a closure copied into every
venue, which is what actually lets `transport_for` be the one factory
*everywhere* instead of everywhere-but-inside-a-metal-container.
(2) `builds_proposed`: one named row→record crossing at the desk's wire door,
so `Desk` speaks `Builds` and the wire speaks rows (no meta-dict bags).
(3) `tests/venue_stub.py`: a Modal stand-in, which makes the `deploy/` files
importable by the suite at all — that is what promise 7's byte-comparison
rides on, and it is a durable capability rather than a test fixture.

**What stayed unproven.** No metal was run (the ADR forbade it of the
implementing session). Named specifically: that a Modal container answers the
new `door` / `door_ask` signature; that the desk's `boot_for` spawn wakes
metal in *another* app; that `deploy/ui.py` over the mount reads fresh — the
SDK-backed reader it replaces was written **because** a mid-scan
`volume.reload()` made runs hop root → subdir and timed `/api/runs` out at 60 s,
and the fix here (reload only at the cache-rebuild boundary, under the rebuild
lock) is reasoned, not measured; and `deploy/concept_steer.py` landed at **460
lines, not ≤ 220** — promise 1 missed. The remainder is a 66-line experiment
docstring, five volume-side functions and six doors: every line is science or a
door, but there is more of both than the promise estimated. Line counts for the
other two: `steer_l4.py` 948 → 555, `stress_fleet.py` 1226 → 881 (the ADR's
Context said 976; it was 1226 by the time this was implemented).

**Samarth's, before this is `Implemented`:** `modal deploy deploy/desk.py`,
then `steer_l4::probe` and `::check` through the one desk, then
`stress_fleet::topology` / `::latejoin` / `::learner_sleep`. And one edit only
this repo's owner makes: **STYLE.md rule 8's tree** names neither
`runner/transports/` (the new region) nor the fact that `runner/sources/` is
now empty. `tests/test_architecture.py` derives its regions from its own table
and enforces `runner/transports/` either way, so nothing is unguarded — the
document is simply behind the tree.
