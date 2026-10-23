"""Small bank-indexed Triton operators; no base-model GEMM is repeated."""

import torch
import triton
import triton.language as tl


@triton.jit
def _steer(Y, V, S, A, O, N: tl.constexpr, D: tl.constexpr,
           Y0: tl.constexpr, Y1: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
           BLOCK: tl.constexpr):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = k // D, k % D
    active = tl.load(A + row, row < N, other=0)
    slot = tl.load(S + row, row < N, other=0)
    value = tl.load(Y + row * Y0 + col * Y1, row < N, other=0)
    delta = tl.load(V + slot * V0 + col * V1, (row < N) & active, other=0)
    tl.store(O + k, value + delta, row < N)


def steer(output, vectors, slots, active):
    result = torch.empty_like(output, memory_format=torch.contiguous_format)
    n, d = output.shape
    _steer[(triton.cdiv(n * d, 256),)](
        output, vectors, slots, active, result, n, d, *output.stride(), *vectors.stride(), 256)
    return result


@triton.jit
def _shrink(X, A, S, ACTIVE, TMP, N: tl.constexpr, I: tl.constexpr, R: tl.constexpr,
            X0: tl.constexpr, X1: tl.constexpr, A0: tl.constexpr,
            A1: tl.constexpr, A2: tl.constexpr, BLOCK: tl.constexpr):
    row, rank = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    active = tl.load(ACTIVE + row)
    slot = tl.load(S + row)
    x = tl.load(X + row * X0 + cols * X1, (cols < I) & active, other=0).to(tl.float32)
    a = tl.load(A + slot * A0 + rank * A1 + cols * A2,
                (cols < I) & active, other=0).to(tl.float32)
    tl.store(TMP + row * R + rank, tl.sum(x * a, 0))


@triton.jit
def _expand(Y, B, S, ACTIVE, TMP, O, D: tl.constexpr, R: tl.constexpr,
            Y0: tl.constexpr, Y1: tl.constexpr, B0: tl.constexpr,
            B1: tl.constexpr, B2: tl.constexpr, RANK: tl.constexpr,
            COLS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    ranks = tl.arange(0, RANK)
    slot = tl.load(S + row)
    active = tl.load(ACTIVE + row)
    tmp = tl.load(TMP + row * R + ranks, ranks < R, other=0)
    b = tl.load(B + slot * B0 + cols[:, None] * B1 + ranks[None, :] * B2,
                (cols[:, None] < D) & (ranks[None, :] < R) & active, other=0).to(tl.float32)
    delta = tl.sum(b * tmp[None, :], 1)
    y = tl.load(Y + row * Y0 + cols * Y1, cols < D, other=0)
    tl.store(O + row * D + cols, y + delta, cols < D)


def low_rank(x, output, a, b, slots, active):
    n, i = x.shape
    d, r = b.shape[1:]
    tmp = torch.empty((n, r), device=output.device, dtype=torch.float32)
    result = torch.empty_like(output, memory_format=torch.contiguous_format)
    _shrink[(n, r)](x, a, slots, active, tmp, n, i, r, *x.stride(), *a.stride(),
                    triton.next_power_of_2(i))
    _expand[(n, triton.cdiv(d, 32))](output, b, slots, active, tmp, result, d, r,
        *output.stride(), *b.stride(), triton.next_power_of_2(r), 32)
    return result
