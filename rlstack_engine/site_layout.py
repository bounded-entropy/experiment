"""Logical projections are views of one physical packed base GEMM."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionSlice:
    path: str
    start: int
    width: int


def projection_slices(path: str, width: int, output_sizes: tuple[int, ...] = ()):
    result = [ProjectionSlice(path, 0, width)]
    parent, _, leaf = path.rpartition(".")
    names = {"qkv_proj": ("q_proj", "k_proj", "v_proj"),
             "gate_up_proj": ("gate_proj", "up_proj")}.get(leaf)
    if names is not None:
        if len(names) != len(output_sizes) or sum(output_sizes) != width:
            raise ValueError(f"unsupported packed projection layout at {path}")
        start = 0
        for name, size in zip(names, output_sizes):
            result.append(ProjectionSlice(f"{parent}.{name}", start, size))
            start += size
    return tuple(result)
