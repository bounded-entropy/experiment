"""THE CONCEPT CAMPAIGN, as a record (ADR 0005): one base, one concept, three
anchor layers, and the science that follows from them — the teacher's
trajectory set, the SFT arms, the on-policy arms and the measurement — with
no Modal in it. A venue (`concept_steer.py`, `concept_burgers.py`) is one
`Campaign` plus the metal and the doors; the science is written here ONCE so
that two campaigns differ in their record and nothing else.

Every method builds a spec the way the venues always did; the module-level
names in `concept_steer.py` delegate to its own campaign, so the pinned rows
in `tests/venue_spec_rows.json` are untouched by this move.
"""

from __future__ import annotations

from dataclasses import dataclass

ENTRY = "v"                     # the bank's one name; `export` copies v@<version>
ALPHA = 0.1                     # nsteer: the injection STARTS at this fraction of ||h_t|| per token
LORA_RANK = 4                   # mlp_lora: a rank-4 delta on one layer's three MLP matrices
ADAPTERS = ("steer", "nsteer", "mlp_lora")
"""The arms' families: a free vector at the boundary, a norm-scaled direction
there (alpha trained, unit direction), and a small LoRA on the MLP of the
same layer — the paper's anchors, three ways of touching one layer."""
ALGOS = ("sft", "opd")
"""How an arm learns: SFT over the teacher's sealed set (ADR 0005), or ON-POLICY
distillation — the student samples its own rollouts and the conditioned
teacher scores those very tokens (opd: sampled-token reverse KL)."""


@dataclass(frozen=True)
class Campaign:
    """One concept-steering campaign, whole: what is trained, on what, at
    which boundaries, and how its members are placed.

    `split` is the placement shape: False is ONE HostSpec with `main` and the
    learner ALTERNATING on one partition (the 32B's, which cannot co-reside
    on 80 GB), True is TWO HostSpecs — the engine and the learner as their
    own placement units, co-resident by fraction on the same device or on
    different ones — so nothing takes turns and the on-policy arms pipeline
    (found 2026-09-05: alternation made a 14-minute wave of a 3-minute one).
    """

    base: str
    hidden: int                 # the boundary's width — a boundary has no shape
    width: int                  # tp for the pool, fsdp for the learner
    anchors: tuple[int, ...]    # the three layers; `resid_pre.<n>` is the stream LEAVING layer n
    concept: str
    subdir: str
    main_gb: float              # the pool's carve, TOTAL across its shards (ADR 0001)
    learner_gb: float           # the learner's, likewise
    split: bool
    lr: float = 5e-3
    microbatch_tokens: int = 4096
    waves: int = 64             # the set: WAVES waves of PER_WAVE trajectories
    per_wave: int = 32
    max_tokens: int = 1024
    teacher_seed: int = 5
    student_seed: int = 11
    opd_seed: int = 13
    measure_seed: int = 3

    # ---- placement ----------------------------------------------------------

    def serving_topology(self):
        """The teacher's: `main` alone, because a generation-only run has no
        learner to alternate with (ADR 0006 Part B)."""
        from rlstack import HostSpec, PoolMember, Topology

        return Topology(hosts=(HostSpec((
            PoolMember("main", tp=self.width, vram_gb=self.main_gb),)),))

    def topology(self):
        """An SFT arm's: the pool (declared, idle — the gate holds the
        steer's boundary against its inventory) and the learner, alternating
        on one partition or as two units (`split`)."""
        from rlstack import HostSpec, LearnerMember, PoolMember, Topology

        main = PoolMember("main", tp=self.width, vram_gb=self.main_gb)
        learner = LearnerMember(fsdp=self.width, vram_gb=self.learner_gb)
        if self.split:
            return Topology(hosts=(HostSpec((main,)), HostSpec((learner,))))
        return Topology(hosts=(HostSpec((main, learner)),))

    def opd_topology(self):
        """The on-policy arm's: the SFT shape plus the TEACHER as its own
        placement unit — the bare base under the hint, scored over the wire.
        A second HostSpec of the same base joins the pool's listing where one
        covers it, which is one engine under two names (the measurement's
        arrangement, declared)."""
        from rlstack import HostSpec, PoolMember, Topology

        teacher = HostSpec((PoolMember("teacher", tp=self.width,
                                       vram_gb=self.main_gb),))
        return Topology(hosts=tuple(self.topology().hosts) + (teacher,))

    # ---- plans --------------------------------------------------------------

    def teacher_rollout_plan(self, task_ids):
        """WAVES waves of PER_WAVE prompts, each sampled ONCE under the
        conditioned teacher: a trajectory set is a set of prompts answered,
        not a group structure — nothing here computes an advantage."""
        from rlstack import GroupPlan, RunPlan, Sample, WavePlan

        def wave(u: int):
            return WavePlan(tuple(
                GroupPlan(task_id, (Sample(task_id, "conditioned_teacher"),))
                for task_id in task_ids[u * self.per_wave:(u + 1) * self.per_wave]))
        return RunPlan(tuple(wave(u) for u in range(self.waves)))

    def sft_train_plan(self, teacher_run: str):
        """Update u trains on the teacher's rollout u, whole — one group per
        wave, this run's own key, the teacher's rows by ref (ADR 0006's
        `store://<run_id>/rollouts/<r>#<i>`)."""
        from rlstack import GroupPlan, Replay, RunPlan, WavePlan

        return RunPlan(tuple(
            WavePlan((GroupPlan(f"distill-{u}", tuple(
                Replay(f"store://{teacher_run}/rollouts/{u}#{i}")
                for i in range(self.per_wave))),))
            for u in range(1, self.waves + 1)))

    def onpolicy_rollout_plan(self, task_ids):
        """The student's OWN waves: WAVES x PER_WAVE prompts, each sampled
        once by the student under `single_turn` — no hint, no advantage; the
        teacher's opinion of those tokens is the post pipeline's, not the
        plan's."""
        from rlstack import GroupPlan, RunPlan, Sample, WavePlan

        def wave(u: int):
            return WavePlan(tuple(
                GroupPlan(task_id, (Sample(task_id, "single_turn"),))
                for task_id in task_ids[u * self.per_wave:(u + 1) * self.per_wave]))
        return RunPlan(tuple(wave(u) for u in range(self.waves)))

    def onpolicy_train_plan(self):
        """Update u trains on this run's OWN rollout u, whole (ADR 0006's
        `self://rollouts/<u>`), one group per wave."""
        from rlstack import RunPlan, WaveRef

        return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                             for u in range(1, self.waves + 1)))

    # ---- the bank -----------------------------------------------------------

    def bank_entry(self, adapter: str, layer: int | str):
        """ONE LAYER, THREE WAYS: the bank's one entry for an arm. `steer` and
        `nsteer` sit at the boundary the paper adds at (`resid_pre.<layer>`,
        the output of model.layers[layer]); `mlp_lora` is a rank-LORA_RANK
        delta on that layer's gate, up and down projections — the same
        layer's own computation, changed a little, instead of its output,
        pushed. `layer` may be a RANGE in the site grammar ("0-8": one
        direction per boundary of the first third, alpha shared) — the
        2026-09-05 ask, "steer the first 1/3 of the layers, the second 1/3,
        and the third 1/3"."""
        from rlstack import lora, nsteer, steer

        if adapter == "steer":
            return steer(f"resid_pre.{layer}", d=self.hidden)
        if adapter == "nsteer":
            return nsteer(f"resid_pre.{layer}", d=self.hidden, alpha=ALPHA)
        if adapter == "mlp_lora":
            return lora(f"layers.{layer}.mlp.*", r=LORA_RANK)
        raise ValueError(f"adapter must be one of {ADAPTERS}; got {adapter!r}")

    # ---- the specs ----------------------------------------------------------

    def _train_ids(self, store, train_tasks: str) -> list[str]:
        from rlstack import load_tasks

        ids = [task.id for task in load_tasks(store, train_tasks)]
        if len(ids) < self.waves * self.per_wave:
            raise ValueError(
                f"{train_tasks} holds {len(ids)} prompts; the plan wants "
                f"{self.waves * self.per_wave}")
        return ids

    def teacher_spec(self, store, train_tasks: str):
        """THE TRAJECTORY SET AS A RUN (ADR 0005, Q4): an EMPTY bank, so
        `main` is the bare base and the hint is the whole of the
        conditioning; no algo, so no Trainer, no ledger, and the extent is
        the rollout plan. Its sealed rollouts are the set and its run_id is
        the set's identity."""
        from rlstack import (
            ExperimentSpec, GenSpec, Plans, PolicySpec, SamplingSpec, Seeds,
            encode,
        )

        ids = self._train_ids(store, train_tasks)
        plan = store.cas_put(encode(self.teacher_rollout_plan(ids)))
        return ExperimentSpec(
            policy=PolicySpec(base=self.base, bank={}),
            gen=GenSpec(envs=("conditioned_teacher",), tasks=(train_tasks,),
                        sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                              max_tokens=self.max_tokens)),
            plans=Plans(train=None, rollout=plan),
            algo=None,
            topology=self.serving_topology(),
            seeds=Seeds(master=self.teacher_seed))

    def student_spec(self, store, teacher_run: str, layer: int | str,
                     adapter: str = "steer"):
        """ONE ARM: the same base with ONE delta at one anchor, SFT over the
        teacher's rows, sampling nothing of its own. The arms share plan
        bytes and differ in the bank alone."""
        from rlstack import (
            AlgoSpec, ExperimentSpec, OptimSpec, Plans, PolicySpec, Schedule,
            Seeds, encode,
        )

        entry = self.bank_entry(adapter, layer)
        plan = store.cas_put(encode(self.sft_train_plan(teacher_run)))
        return ExperimentSpec(
            policy=PolicySpec(base=self.base, bank={ENTRY: entry}),
            gen=None,
            plans=Plans(train=plan, rollout=None),
            algo=AlgoSpec(loss="sft", post=(),
                          optim=OptimSpec("adamw", lr=self.lr),
                          schedule=Schedule(microbatch_tokens=self.microbatch_tokens,
                                            max_policy_lag=0)),
            topology=self.topology(),
            seeds=Seeds(master=self.student_seed))

    def opd_spec(self, store, train_tasks: str, layer: int | str,
                 adapter: str = "nsteer"):
        """ON-POLICY DISTILLATION, one arm: the student samples the train
        prompts under its own delta (no hint), the conditioned teacher scores
        each sampled token under the hint through the `teacher` pool, and
        `opd` — the sampled-token reverse KL — pulls the student toward the
        teacher on the student's own distribution. No sealed set is
        replayed; the extent is this run's own waves."""
        from rlstack import (
            AlgoSpec, ExperimentSpec, GenSpec, OptimSpec, Plans, PolicySpec,
            SamplingSpec, Schedule, Seeds, encode,
        )

        ids = self._train_ids(store, train_tasks)
        rollout = store.cas_put(encode(self.onpolicy_rollout_plan(ids)))
        train = store.cas_put(encode(self.onpolicy_train_plan()))
        return ExperimentSpec(
            policy=PolicySpec(base=self.base,
                              bank={ENTRY: self.bank_entry(adapter, layer)}),
            gen=GenSpec(envs=("single_turn",), tasks=(train_tasks,),
                        sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                              max_tokens=self.max_tokens)),
            plans=Plans(train=train, rollout=rollout),
            algo=AlgoSpec(loss="opd", post=("conditioned_teacher_logprobs",),
                          optim=OptimSpec("adamw", lr=self.lr),
                          schedule=Schedule(microbatch_tokens=self.microbatch_tokens,
                                            max_policy_lag=0)),
            topology=self.opd_topology(),
            seeds=Seeds(master=self.opd_seed))

    def probe_spec(self, store, heldout_tasks: str, parent_run: str | None,
                   version: int, layer: int | str, adapter: str = "nsteer",
                   waves: int = 2, per_wave: int = 8, max_tokens: int = 256):
        """A PROBE: the student answers a few held-out prompts under a
        trained delta, sampled from the store — no hint, no learner, no
        loss; the sealed rollouts are the point ("do they actually output
        anything relevant to burgers"). The delta is the parent run's blob
        at `version`, served FROZEN through a WarmStart (ADR 0006 Part B: a
        run with no Trainer serves its parent's sealed payload as version 0
        for life). `parent_run` None is the bare base, the baseline."""
        import dataclasses

        from rlstack import (
            ExperimentSpec, GenSpec, Plans, PolicySpec, SamplingSpec, Seeds,
            WarmStart, encode, load_tasks,
        )

        ids = [task.id for task in load_tasks(store, heldout_tasks)][:waves * per_wave]
        if len(ids) < waves * per_wave:
            raise ValueError(f"{heldout_tasks} holds {len(ids)} prompts; the "
                             f"probe wants {waves * per_wave}")
        campaign = dataclasses.replace(self, waves=waves, per_wave=per_wave)
        plan = store.cas_put(encode(campaign.onpolicy_rollout_plan(ids)))
        bank = ({} if parent_run is None else
                {ENTRY: dataclasses.replace(self.bank_entry(adapter, layer),
                                            trainable=False)})
        return ExperimentSpec(
            policy=PolicySpec(base=self.base, bank=bank),
            gen=GenSpec(envs=("single_turn",), tasks=(heldout_tasks,),
                        sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                              max_tokens=max_tokens)),
            plans=Plans(train=None, rollout=plan),
            algo=None,
            init=(None if parent_run is None else
                  WarmStart(policy=f"store://{parent_run}@{version}", optim="fresh")),
            topology=self.serving_topology(),
            seeds=Seeds(master=21))

    def the_measurement(self, task_ids):
        """The distillation number, OUTSIDE the run (#70): the student
        answers held-out prompts under its own delta, the conditioned
        teacher scores those very tokens, and `reverse_kl` is how many nats
        apart they are."""
        from rlstack import Measurement

        return Measurement(
            name="distill", env="single_turn", task_ids=tuple(task_ids),
            samples=1, every=8,
            post=("conditioned_teacher_logprobs", "reverse_kl"),
            seed=self.measure_seed, temperature=1.0, max_tokens=self.max_tokens)
