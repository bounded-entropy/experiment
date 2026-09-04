"""The runner: the substrate that drives both worlds as daemons on a blackboard.

The map, by responsibility — read top to bottom:

    interfaces.py   the seams: Engine / Learner protocols + what crosses them
                    (TokenEvent, FinishEvent, TrainStats, Emitted)
    loop.py         run_experiment(_async): Phase 0 identity, Phase 1 setup,
                    needs_of + plan_daemons — the only orchestration. A run is
                    a set of daemon NEEDS with the resources each admits; the
                    experiment is the set read off a spec (ADR 0006 Part B)
    daemons/        one file per GPU responsibility (generator / trainer /
                    scorer), all the same four beats: await condition,
                    admit residents, work, write + notify — each present
                    exactly where its need is
    signals.py      the LOGICAL half of the blackboard: awaitable predicates
                    over the store — daemons never call each other
    arbiter.py      the PHYSICAL half: the Arbiter owns admission to the
                    metal — object-keyed residents, exclusive groups from
                    a multi-member HostSpec, sticky drain-until-blocked
    traffic.py      what travels to pools: EnginePoolClient (sample +
                    score) → episode seal (run_episode) → sealed wave
                    (collect_wave)
    post.py         EXECUTES the declared post pipeline per group (the
                    processors themselves are declared in training/post/)
    host.py         a host: an atomic purposed partition (Partition +
                    Regimes) owning its engines, its one learner, its arbiter
                    and its journal; submit binds / fits / attests / runs
    desk.py         the desk (listings, metal, placement over them: join →
                    carve → acquire; the recipe it DECLARES per metal, and
                    the guards over release and decommission) and
                    MetalService, the metal-side books that boot BARE and
                    spawn residents from the recipe a carve delivers
    residents.py    a resident is a process (ADR 0002): the build records a
                    venue declares, the universal builders, the door and its
                    frames, Resident.spawn / in_process / stop, the ladder
    remote.py       the wire: HostService admits pool verbs — and, since ADR
                    0006 Part A, learner verbs — at its own arbiter and
                    forwards through the resident's proxy; EngineService /
                    LearnerService are the resident's end; RemotePool /
                    RemoteLearner are the whole protocols over a Transport,
                    across hosts as readily as inside one. Also THE ADDRESS
                    GRAMMAR and the one transport factory (ADR 0007):
                    parse_address / transport_for, with the in-process
                    switchboard consulted first
    assemble.py     a planned wave becomes real: sample it, or take it
    refs.py         where an already-sealed trajectory lives: this run's own
                    rollouts, another run's waves or rollouts, a cas file
    seeds.py        the seed tree: derive(master, *path)
    meters.py       the emission plane: the TrafficMeter engines and the
                    arbiter count into (drained once per stats tick), the
                    UpdateClock the Trainer laps, and the HostJournal door
                    they reach — wall clock, so host journals only
    engines/        real inference metal (vllm_engine), heavy imports,
    learners/       real training metal (torch_learner),
    transports/     real wire substrates (modal_cls) — all three import
                    their substrate at module scope and are therefore
                    imported LAZILY, never from the package root (STYLE
                    rule 7, pinned by tests/test_architecture.py)
    fakes.py        FakeEngine / FakeLearner / FakeAdapter: deterministic
                    stand-ins behind the same protocols (metal, and the one
                    stdlib adapter type), for tests and dry runs

This is the one package allowed to import both worlds.
"""

from rlstack.runner import (  # noqa: F401
    meters, interfaces, seeds, traffic, signals, arbiter, post,
    daemons, loop, host, fakes,
)
