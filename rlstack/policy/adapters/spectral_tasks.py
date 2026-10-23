"""One spectral decoder with independently routed task posteriors.

Trainer-only: supervised records supply the task index and a Gaussian draw.
No task ID enters the decoder. Replay rows, optimizer groups and checkpoints
use the existing adapter contract.
"""

from rlstack.policy.adapters.definition import Operation, SiteKind, builtin
from rlstack.policy.adapters.base import AdapterType, adapter_type
from rlstack.policy.adapters.spectral_latent import SpectralLatent

TASK_RECORD = "spectral_task"
EPS_RECORD = "slatent_eps"


@adapter_type("spectral_tasks")
class SpectralTasks(AdapterType):
    definition = builtin(Operation.SPECTRAL_TASKS, SiteKind.LINEAR, 'spectral_tasks')
    records = (TASK_RECORD, EPS_RECORD)
    provides = SpectralLatent.provides | frozenset({"factual_beta"})

    def site_ok(self, meta):
        return meta.has_weight


    def provide(self, params):
        from rlstack.policy.adapters.spectral_tasks_torch import provide
        return provide(params)

    def param_groups(self, params):
        from rlstack.policy.adapters.spectral_tasks_torch import param_groups
        return param_groups(params)
