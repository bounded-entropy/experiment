# The latent → LoRA map, and the two bases

How a plora entry turns one latent draw into an ordinary rank-k LoRA, and
exactly what the `basis="random"` (no-SVD) arm changes. Code:
`rlstack/policy/adapters/plora_torch.py` (the map),
`rlstack/policy/adapters/plora_factors.py` (the two frozen-basis recipes).

## The shared pipeline (identical in both arms)

```
eps  (sealed at rollout, one per trajectory)
 │
 ▼
z = mu + exp(log_std) * eps          reparameterized with the CURRENT posterior
 │
 ▼
h = trunk(z)                          bias-free hypernet: enter + 2 residual blocks
 │
 ▼
C_s = head_s(h)                       one zero-init head per site  →  k×k core
 │
 ▼
Delta_s = U_s · C_s · A_s             the delta at site s;  U_s, A_s FROZEN
```

- **One latent per trajectory.** The engine draws `eps ~ N(0, I)` per request
  and seals it into `turn_extras`. At replay the trainer recomputes
  `z = mu + exp(log_std) * eps` with the *current* posterior, so the
  reparameterized gradient reaches `mu` and `log_std` through a draw that
  already happened. (Recording `z` instead would freeze the posterior out of
  its own gradient — that is why the *noise* is the sealed fact.)
- **What trains:** `mu`, `log_std`, the trunk, the per-site heads.
  **What never moves:** `U_s` (shape `out × k`) and `A_s` (shape `k × in`).
- **Serving through punica:** ensemble member m materializes as a plain
  rank-k peft adapter with

  ```
  lora_A = A_s               (identical at every version)
  lora_B = U_s · C_s(z_m)    (the only moving part)
  ```

  so the ensemble is just E ordinary LoRAs; nothing about the hypernet
  survives into the served format.
- **The KL** `KL(q || p)` with `q = N(mu, diag(exp(log_std)^2))` and
  `p = N(0, prior_std^2 I)` is a provided tensor in both arms, priced the
  same way by `grpo_latent_kl_gated`. Zero at init (mu = 0,
  log_std = log(prior_std)), and every core is zero at init (zero heads), so
  version 0 IS the base model in both arms.

## Arm 1 — `basis="svd"`: the weight's own frame

Each matched weight is factored once:

```
M  ≈  U_k · Sigma_k · V_kᵀ           (top-k truncated SVD)

U_s = U_k                            out × k, orthonormal columns
A_s = Sigma_k · V_kᵀ                 k × in   (Sigma absorbed into A,
                                     because peft has no third factor)
```

The core steers **inside the weight's own top-k singular subspace**: the
delta's column space is the top left-singular directions, its row space the
top right-singular directions, scaled by the singular values.

## Arm 2 — `basis="random"`: a scale-matched random frame

Per site, draw a seeded Gaussian, QR it, fix the sign freedom
(make `diag(R) > 0` — the QR analog of the SVD recipe's sign rule):

```
U' = orthonormal(out × k)            random, from seed sha(basis_seed:path:u)
V' = orthonormal(in  × k)            random, from seed sha(basis_seed:path:v)

U_s = U'
A_s = Sigma_k · V'ᵀ                  the SAME top-k singular values as arm 1
```

`Sigma_k` is still computed from the weight (same Gram-matrix `eigh`); only
the **directions** are randomized.

## Why keep Sigma_k in the random arm

With `U` and `V` orthonormal in both arms, the size of the delta a given core
produces scales with the singular values:

```
|| U · C · A ||_F  =  || C · Sigma_k ||-ish        (orthonormal frames
                                                    preserve norms)
```

So keeping `Sigma_k` means **a unit core moves the weights by the same
magnitude in both arms**. Drop it and the random arm would differ in
effective step size *and* direction at once — the ablation would be
confounded. As built, the two arms share the latent, the hypernet, the cores,
the ensemble, the KL, and the delta magnitudes; the ONLY difference is
**where the rank-k delta can point**: the top singular subspace vs a
randomly-oriented k-dimensional subspace.

## Determinism and identity

- The random frame is a pure function of `(basis_seed, site path)` — no
  global RNG, no eigenvector ambiguity beyond the canonicalized column signs.
- `basis_seed` lives in the spec's init dict, so it hashes into `run_id`.
- Both sides of the bridge agree by construction: the trainer *recomputes*
  `(U', A)` from the weight it already holds at install; the engine reads the
  deploy-built factors artifact. The artifact is stamped with its recipe id —
  `gram-eigh-fp32-canonical-sign-v1` for SVD,
  `random-orthonormal-svd-scale-v1` for random — and a reader refuses an id
  it does not recognize, so the two coordinate systems can never be silently
  interchanged.
- Same precision story as the SVD arm: the artifact stores bf16, the trainer
  recomputes fp32, and the difference sits under the bf16 kernel floor that
  `logprob_gap` already watches.
