"""rlstack — thin RL wrapper, Phase B1.

Canon: SPEC.md. Identity is computed (I3); training consumes only sealed data (I1).

Layout mirrors the architecture:
  spec/       the contract (declarative values, identity, submit gate)
  registry.py the index: name -> typed record (declaration + compute)
  client.py   the sampling interface both worlds type against
  policy/     the bridge (I2): sites + adapters/ + bundle compile
  inference/  the sampling world (Rollout, environments/)
  training/   the gradient world (post/ pipeline, losses)
  data/       the membrane (trajectory→group→wave, flatten+pack, stores/)
  runner/     the substrate (Engine/Learner protocols, the loop, fakes)
"""

__version__ = "0.0.1"

from rlstack.spec.canonical import canonical_json, content_hash, run_id
from rlstack.spec.specs import (
    AdapterSpec, AlgoSpec, BackendProfile, PoolMember, EvalSpec, ExperimentSpec,
    GenSpec, GpuConfig, GpuSet, GpuGroup, LearnerMember, OptimSpec, PolicySpec,
    SamplingSpec, Schedule, Seeds, TrajectorySource, WarmStart,
    attn_bias, gpus, learner, lora, pool, soft_prompt,
)
from rlstack.registry import (
    ADAPTERS, ENVS, LOSSES, POST,
    LossDef, Probe, Ref, Registry, Teacher,
    code_hashes, loss, source_hash,
)
from rlstack.client import SampleClient
from rlstack.policy.siteschema import SiteMeta, SiteSchema, fake_qwen_schema, resolve
from rlstack.policy.adapters import Adapter, AdapterDef, Mechanism, adapter
from rlstack.inference.rollout import Rollout
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
    DataError, Group, Message, Role, Task, Trajectory, Turn, Wave,
    trajectory_from_row, trajectory_to_row, wave_from_rows, wave_to_rows,
)
from rlstack.data.flatten import Flat, TokenBatch, broadcast, flatten, pack
from rlstack.data.stores import (
    LedgerError, LocalStore, ManifestMismatch, ModalVolumeStore, RunHandle,
    Store, StoreError, bump,
)
from rlstack.policy.compile import Bundle, compile_bundle, group_by_mechanism
from rlstack.runner.interfaces import (
    Emitted, Engine, FinishEvent, Learner, TokenEvent, TrainStats,
)
from rlstack.runner.seeds import derive
from rlstack.runner.sampling import (
    EngineSampleClient, Routes, collect_wave, load_tasks,
)
from rlstack.runner.post import run_pipeline
from rlstack.runner.signals import RunSignals
from rlstack.runner.lease import (
    ENGINE, LEARNER, ExclusiveLease, Lease, LeaseMap, OpenLease, leases_for,
)
from rlstack.runner.sources import (
    LiveFeed, ReplayFeed, StaticFeed, WaveFeed, feed_for,
)
from rlstack.runner.daemons import Daemon, Evaluator, Generator, Trainer
from rlstack.runner.loop import RunReport, plan_daemons, run_experiment
from rlstack.runner.fakes import FakeEngine, FakeLearner
