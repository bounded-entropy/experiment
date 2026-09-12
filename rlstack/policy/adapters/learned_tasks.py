"""Shared learned directions with independently sampled task posteriors."""

from rlstack.policy.adapters.base import AdapterType, adapter_type
from rlstack.policy.adapters.spectral_tasks import EPS_RECORD, TASK_RECORD


@adapter_type("learned_tasks")
class LearnedTasks(AdapterType):
    records = (TASK_RECORD, EPS_RECORD)
    provides = frozenset({"latent_kl", "factual_beta", "posterior_std"})

    def site_ok(self, meta):
        return meta.has_weight

    def params(self, sites, init):
        from rlstack.policy.adapters.learned_tasks_torch import build
        return build(sites, init)

    def install_replay(self, model, params, sites):
        from rlstack.policy.adapters.learned_tasks_torch import install
        install(model, params)

    def uninstall_replay(self, model, params, sites):
        from rlstack.policy.adapters.learned_tasks_torch import uninstall
        uninstall(model, params)

    def provide(self, params):
        from rlstack.policy.adapters.learned_tasks_torch import provide
        return provide(params)

    def param_groups(self, params):
        from rlstack.policy.adapters.learned_tasks_torch import param_groups
        return param_groups(params)

    def emit(self, params):
        from rlstack.policy.adapters.learned_tasks_torch import emit
        return emit(params)

    def load(self, params, payload):
        from rlstack.policy.adapters.learned_tasks_torch import load
        load(params, payload)
