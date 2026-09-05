"""rlstack — a thin, high-throughput RL harness for LLMs.

An experiment is one spec value submitted to a host; its identity is computed
from that value and the source of everything it names, never typed (I3). The
folders are the architecture:

  spec/            the contract — declarative values, identity, the flow
                   graph, the submit gate
  registry.py      the index: name -> typed record (declaration + compute)
  client.py        the pool interface both worlds may type against policy/
  the bridge: sites + adapters/ (one file per adapter type per side)
                   + bundle compile
  inference/       the sampling world — Rollout, environments/
  training/        the gradient world — post/ processors, losses/
  data/            the membrane — trajectory→group→wave, flatten+pack, stores/
  runner/          the substrate — Engine/Learner, the fleet and the wire, the
                   blackboard, daemons/, engines/, learners/, fakes
  observe/         read-only derivations over stores and journals
  rlstack_engine/  (sibling package) what ships in the ENGINE image
"""

__version__ = "0.0.1"

from rlstack.spec.canonical import canonical_json, content_hash, run_id
from rlstack.spec.flow import FlowGraph, FlowNode, flow_graph
from rlstack.data.plan import (
    Derive, GroupPlan, Replay, RunPlan, Sample, WavePlan, WaveRef, decode,
    encode,
)
from rlstack.spec.specs import (
    AdapterSpec, AlgoSpec, BackendProfile, PoolMember, ExperimentSpec,
    GenSpec, Topology, HostSpec, LearnerMember, OptimSpec, PolicySpec,
    Plans, SamplingSpec, Schedule, Seeds, WarmStart,
    attn_bias, learner, lora, nsteer, plora, pool, soft_prompt, steer,
)
from rlstack.registry import (
    ADAPTER_TYPES, ENVS, LOSSES, POST,
    LossDef, Registry,
    code_hashes, loss, source_hash,
)
from rlstack.client import PoolClient
from rlstack.policy.siteschema import SiteMeta, SiteSchema, fake_qwen_schema, resolve
from rlstack.policy.adapters import (
    AdapterType, AdapterTypeDef, Directive, Mechanism, adapter_type,
)
from rlstack.policy.adapters.steer import SteerWindow
from rlstack.inference.rollout import Rollout
from rlstack.inference import makers as _makers   # registers task makers
from rlstack.inference.environments import (
    Environment, EnvironmentDef, environment,       # registers builtin envs
)
from rlstack.training.post import (
    PostDef, PostProcessor, postprocessor,          # registers builtin postprocessors
)
from rlstack.training.post.grpo_advantage import zscore
from rlstack.training import losses as _losses      # registers builtin losses
from rlstack.spec.validate import (
    SpecError, ValidationIssue, check_sites_reachable_on, site_space,
    validate, validate_or_raise,
)
from rlstack.data.trajectory import (
    DataError, Group, Message, Role, Task, Trajectory, Turn, Wave, hint_for,
    trajectory_from_row, trajectory_to_row, wave_from_rows, wave_to_rows,
)
from rlstack.data.flatten import Flat, TokenBatch, broadcast, flatten, pack
from rlstack.data.tasks import load_tasks, split_tasks, write_tasks
from rlstack.data.stores import (
    DEFAULT_RETENTION, KeepRestorable, LedgerError, LocalStore,
    ManifestMismatch, ModalVolumeStore, RetentionPolicy, RunHandle,
    RunProgress, Store, StoreAddress, StoreError, Swept, bump, open_store,
    run_done, run_progress,
)
from rlstack.policy.compile import (
    Bundle, compile_bundle, group_by_adapter_type, group_by_mechanism,
)
from rlstack.runner.interfaces import (
    Emitted, Engine, EntryInstall, FinishEvent, Learner, OptimSettings,
    Parameterization, TokenEvent, TrainStats,
)
from rlstack.runner.seeds import derive
from rlstack.runner.traffic import EnginePoolClient, Routes, load_task_sets
from rlstack.runner.assemble import realize, rollouts_needed, sample_wave
from rlstack.runner.refs import RefReader
from rlstack.runner.post import run_pipeline
from rlstack.runner.signals import RunSignals
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.daemons import Daemon, Generator, Scorer, Trainer
from rlstack.runner.host import (
    Host, HostError, Partition, Regime, Tenancy,
)
from rlstack.runner.remote import (
    EngineService, HostService, LearnerService, LocalTransport, RemoteLearner,
    RemotePool, Transport,
)
from rlstack.runner.residents import (
    Builds, EngineBuild, FakeEngineBuild, FakeLearnerBuild, LearnerBuild,
    Resident, ResidentBirth, ResidentError, Teardown,
)
from rlstack.runner.campaign import Campaigns, demands_of
from rlstack.runner.desk import (
    Demand, Desk, DeskError, Metal, demand_rows, demands_from,
    fraction_for_gb,
)
from rlstack.runner.measure import Measurement, measure_run
from rlstack.runner.loop import (
    DaemonNeed, RunReport, experiment_identity, needs_of, parameterization_of,
    plan_daemons, run_experiment,
)
from rlstack.runner.fakes import FakeAdapter, FakeEngine, FakeLearner
