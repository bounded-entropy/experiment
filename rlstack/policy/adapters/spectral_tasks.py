"""One spectral decoder with independently routed task posteriors.

Trainer-only: supervised records supply the task index and a Gaussian draw.
No task ID enters the decoder. Replay rows, optimizer groups and checkpoints
use the existing adapter contract.
"""

from rlstack.policy.adapters.base import AdapterType, adapter_type
from rlstack.policy.adapters.spectral_latent import SpectralLatent

TASK_RECORD = "spectral_task"
EPS_RECORD = "slatent_eps"


@adapter_type("spectral_tasks")
class SpectralTasks(AdapterType):
    records = (TASK_RECORD, EPS_RECORD)
    provides = SpectralLatent.provides | frozenset({"factual_beta"})

    def site_ok(self, meta):
        return meta.has_weight

    def params(self, sites, init):
        from rlstack.policy.adapters.spectral_tasks_torch import build
        return build(sites, init)

    def install_replay(self, model, params, sites):
        from rlstack.policy.adapters.spectral_tasks_torch import install
        install(model, params)

    def uninstall_replay(self, model, params, sites):
        from rlstack.policy.adapters.spectral_tasks_torch import uninstall
        uninstall(model, params)

    def provide(self, params):
        from rlstack.policy.adapters.spectral_tasks_torch import provide
        return provide(params)

    def param_groups(self, params):
        from rlstack.policy.adapters.spectral_tasks_torch import param_groups
        return param_groups(params)

    def emit(self, params):
        from rlstack.policy.adapters.spectral_tasks_torch import emit
        return emit(params)

    def load(self, params, payload):
        from rlstack.policy.adapters.spectral_tasks_torch import load
        load(params, payload)
