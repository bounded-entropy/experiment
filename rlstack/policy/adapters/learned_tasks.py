"""Shared learned directions with independently sampled task posteriors."""

from rlstack.policy.adapters.definition import Operation, SiteKind, builtin
from rlstack.policy.adapters.base import AdapterType, adapter_type
from rlstack.policy.adapters.spectral_tasks import EPS_RECORD, TASK_RECORD


@adapter_type("learned_tasks")
class LearnedTasks(AdapterType):
    definition = builtin(Operation.LEARNED_TASKS, SiteKind.LINEAR, 'learned_tasks')
    records = (TASK_RECORD, EPS_RECORD)
    provides = frozenset({"latent_kl", "factual_beta", "posterior_std"})

    def site_ok(self, meta):
        return meta.has_weight


    def provide(self, params):
        from rlstack.policy.adapters.learned_tasks_torch import provide
        return provide(params)

    def param_groups(self, params):
        from rlstack.policy.adapters.learned_tasks_torch import param_groups
        return param_groups(params)
