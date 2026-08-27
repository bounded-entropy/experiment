"""The runner: the substrate that drives both worlds as daemons on a blackboard.

The map, by responsibility — read top to bottom:

    interfaces.py   the seams: Engine / Learner protocols + what crosses them
                    (TokenEvent, FinishEvent, TrainStats, Emitted)
    loop.py         run_experiment(_async): Phase 0 identity, Phase 1 setup,
                    plan_daemons — the only orchestration
    daemons/        one file per GPU responsibility (generator / trainer /
                    evaluator), all the same four beats: await condition,
                    hold lease, work, write + notify
    signals.py      the LOGICAL half of the blackboard: awaitable predicates
                    over the store — roles never call each other
    lease.py        the PHYSICAL half: who may occupy the metal
                    (OpenLease / ExclusiveLease, from GpuGroup.sharing)
    sampling.py     token stream → Turn (EngineSampleClient) → episode seal
                    (run_episode) → sealed wave (collect_wave)
    post.py         EXECUTES the declared post pipeline per group (the
                    processors themselves are declared in training/post/)
    sources/        WaveFeed: where the trainer's waves come from
                    (live / replay / static), one file each
    seeds.py        the seed tree: derive(master, *path)
    engines/        real inference metal (vllm_engine), heavy imports,
    learners/       real training metal (torch_learner) — import lazily
    fakes.py        FakeEngine / FakeLearner: deterministic stand-ins behind
                    the same protocols, for tests and dry runs

This is the one package allowed to import both worlds.
"""

from rlstack.runner import (  # noqa: F401
    interfaces, seeds, sampling, signals, lease, sources, post, daemons,
    loop, fakes,
)
