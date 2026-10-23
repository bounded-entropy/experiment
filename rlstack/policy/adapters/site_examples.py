"""Two adapters authored entirely as shared state/operation declarations.

site_steer uses init={"d": hidden_width}; site_lora16 has fixed rank 16.
These leave the native lora and legacy steering codecs unchanged.
"""

from rlstack.policy.adapters.base import AdapterType, adapter_type
from rlstack.policy.adapters.definition import (
    AdapterDefinition, Dim, Init, Input, Operation, Parameter, SiteKind, SiteProgram, Symbol,
)


@adapter_type("site_steer")
class SiteSteer(AdapterType):
    definition = AdapterDefinition(
        Operation.CUSTOM, SiteKind.RESIDUAL,
        parameters=(Parameter("direction", (Dim.OUTPUT,)),),
        program=SiteProgram(
            Symbol("rlstack.policy.adapters.site_math_torch", "steer_forward"),
            Symbol("rlstack.policy.adapters.site_math_torch", "steer_reference"),
            helpers=(Symbol("rlstack.policy.adapters.site_math_triton", "steer"),)))


@adapter_type("site_lora16")
class SiteLora16(AdapterType):
    definition = AdapterDefinition(
        Operation.CUSTOM, SiteKind.LINEAR,
        parameters=(Parameter("A", (16, Dim.INPUT), Init.NORMAL),
                    Parameter("B", (Dim.OUTPUT, 16))),
        program=SiteProgram(
            Symbol("rlstack.policy.adapters.site_math_torch", "low_rank_forward"),
            Symbol("rlstack.policy.adapters.site_math_torch", "low_rank_reference"),
            inputs=(Input.INPUT, Input.OUTPUT),
            helpers=(Symbol("rlstack.policy.adapters.site_math_triton", "low_rank"),)))
