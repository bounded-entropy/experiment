"""The runner: the substrate that drives both worlds as daemons on a blackboard.

The map, by responsibility — read top to bottom:

    interfaces.py   the seams: Engine / Learner protocols + what crosses them
                    (TokenEvent, FinishEvent, TrainStats, Emitted)
    loop.py         run_experiment(_async): Phase 0 identity, Phase 1 setup,
                    plan_daemons — the only orchestration
    daemons/        one file per GPU responsibility (generator / trainer /
                    evaluator), all the same four beats: await condition,
                    admit residents, work, write + notify
    signals.py      the LOGICAL half of the blackboard: awaitable predicates
                    over the store — daemons never call each other
    arbiter.py      the PHYSICAL half: the GpuArbiter owns admission to the
                    metal — object-keyed residents, exclusive groups from
                    GpuGroup.sharing="sleep", sticky drain-until-blocked
    traffic.py      what travels to pools: EnginePoolClient (sample +
                    score) → episode seal (run_episode) → sealed wave
                    (collect_wave)
    post.py         EXECUTES the declared post pipeline per group (the
                    processors themselves are declared in training/post/)
    host.py         a host: an atomic purposed partition (Partition +
                    Regimes) owning its engines, its one learner, its arbiter
                    and its journal; submit binds / fits / attests / runs
    fleet.py        the inventory of Metal and hosts, and the placement
                    ladder over them: join → carve → acquire
    remote.py       the wire: HostService serves pool verbs on its own metal
                    under its own arbiter, RemotePool is the whole Engine
                    protocol over a Transport
    sources/        WaveFeed: where the trainer's waves come from
                    (live / replay / static), one file each
    seeds.py        the seed tree: derive(master, *path)
    meters.py       the emission plane: the TrafficMeter engines and the
                    arbiter count into (drained once per stats tick), the
                    UpdateClock the Trainer laps, and the HostJournal door
                    they reach — wall clock, so host journals only
    engines/        real inference metal (vllm_engine), heavy imports,
    learners/       real training metal (torch_learner) — import lazily
    fakes.py        FakeEngine / FakeLearner: deterministic stand-ins behind
                    the same protocols, for tests and dry runs

This is the one package allowed to import both worlds.
"""

from rlstack.runner import (  # noqa: F401
    meters, interfaces, seeds, traffic, signals, arbiter, sources, post,
    daemons, loop, host, fakes,
)
