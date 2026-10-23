"""Reference math and lazy GPU entry points for declaration-only examples."""

import torch


def steer_reference(tensors, parameters, tokens):
    delta = parameters[0][tokens.slot]
    return (tensors.output + torch.where(tokens.active[:, None], delta, 0)).to(tensors.output.dtype)


def steer_forward(tensors, parameters, tokens):
    from rlstack.policy.adapters.site_math_triton import steer
    return steer(tensors.output, parameters[0], tokens.slot, tokens.active)


def low_rank_reference(tensors, parameters, tokens):
    a, b = (bank[tokens.slot] for bank in parameters)
    x = tensors.input.to(a.dtype)
    delta = torch.bmm(b, torch.bmm(a, x.unsqueeze(-1))).squeeze(-1)
    return (tensors.output + torch.where(tokens.active[:, None], delta, 0)).to(tensors.output.dtype)


def low_rank_forward(tensors, parameters, tokens):
    from rlstack.policy.adapters.site_math_triton import low_rank
    return low_rank(tensors.input, tensors.output, *parameters, tokens.slot, tokens.active)
