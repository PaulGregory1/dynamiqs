"""
Implicitly Restarted Arnoldi Method (IRAM) — JAX, fully jittable.

Finds the eigenpair with largest-magnitude eigenvalue.

Key insight: even when only nev=1 eigenvalue is wanted, the implicit restart
must retain k > nev vectors ("guard vectors") so that the compressed
factorisation carries a richer subspace than just the dominant Ritz vector.
With k=1 the IRAM degenerates to naive Ritz-vector restart.  ARPACK uses
k = max(2*nev, nev+2) by default; we expose k as a parameter (default
min(2*nev+1, m-1) to guarantee p >= 1 shift).

Ref: D.C. Sorensen, SIAM J. Matrix Anal. Appl., 13(1), 1992.
     R.B. Lehoucq & D.C. Sorensen, SIAM J. Matrix Anal. Appl., 17(4), 1996.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array


# ── helpers ───────────────────────────────────────────────────────────


def _as(x, dtype):
    return jnp.asarray(x, dtype=dtype)


def _safe_norm(v: Array, eps: Array) -> Array:
    return jnp.maximum(jnp.linalg.norm(v), eps)


# ── Givens-based implicit QR shift ──────────────────────────────────


def _apply_one_shift(H, V, m, mu):
    """
    Apply one implicit QR shift μ to an m-step Arnoldi factorisation.

    Computes the QR factorisation of (H - μI) = QR via Givens rotations,
    then forms  H ← Q^H H Q = RQ + μI  and  V ← V_m Q  (row-stored).

    Returns (H_new, V_new, Q).
    """
    dtype = H.dtype
    Hs = H - mu * jnp.eye(m, dtype=dtype)

    def givens_step(carry, i):
        Hs, Q = carry
        a, b = Hs[i, i], Hs[i + 1, i]

        r = jnp.sqrt(jnp.abs(a) ** 2 + jnp.abs(b) ** 2)
        safe_r = jnp.where(r > 0, r, _as(1.0, jnp.real(a).dtype))
        c = jnp.where(r > 0, a / safe_r, _as(1.0, dtype))
        s = jnp.where(r > 0, b / safe_r, _as(0.0, dtype))

        # G^H on rows i, i+1 (left multiply on Hs)
        ri = jnp.conj(c) * Hs[i] + jnp.conj(s) * Hs[i + 1]
        ri1 = -s * Hs[i] + c * Hs[i + 1]
        Hs = Hs.at[i].set(ri)
        Hs = Hs.at[i + 1].set(ri1)

        # Q ← Q @ G  (right multiply)
        # G = [[c, -conj(s)], [s, conj(c)]]
        qi = Q[:, i] * c + Q[:, i + 1] * s
        qi1 = Q[:, i] * (-jnp.conj(s)) + Q[:, i + 1] * jnp.conj(c)
        Q = Q.at[:, i].set(qi)
        Q = Q.at[:, i + 1].set(qi1)

        return (Hs, Q), None

    Q0 = jnp.eye(m, dtype=dtype)
    (R, Q), _ = jax.lax.scan(givens_step, (Hs, Q0), jnp.arange(m - 1))

    H_new = R @ Q + mu * jnp.eye(m, dtype=dtype)
    # V_m_new = V_m Q  (math cols), V row-stored ⇒ V_new = Q.T @ V
    V_new = Q.T @ V

    return H_new, V_new, Q


def _apply_shifts(H, V, f_vec, m, k, shifts):
    """
    Apply p = m - k implicit QR shifts to compress the factorisation
    from size m to size k.

    Returns (H_new, V_new, f_new) where f_new is the new residual
    vector coupling the k-step factorisation to the next extension.
    """
    dtype = H.dtype
    p = m - k

    def body(carry, j):
        H, V, Q_acc = carry
        H, V, Qj = _apply_one_shift(H, V, m, shifts[j])
        return (H, V, Q_acc @ Qj), None

    Q0 = jnp.eye(m, dtype=dtype)
    (H, V, Q_acc), _ = jax.lax.scan(body, (H, V, Q0), jnp.arange(p))

    # New residual:  f_new = β_{k+1} v_{k+1}^+ + σ_k f
    # where β_{k+1} = H[k, k-1] after shifts, σ_k = Q_acc[m-1, k-1]
    beta_k = H[k, k - 1]
    sigma_k = Q_acc[m - 1, k - 1]
    f_new = beta_k * V[k] + sigma_k * f_vec

    return H, V, f_new


# ── info dataclass (pytree-registered) ───────────────────────────────


@jax.tree_util.register_dataclass
@dataclass
class ArnoldiInfoJit:
    eigval: Array
    ritz_residual: Array
    n_cycles: Array
    n_inner_iters: Array
    metric_hist_cycles: Array
    ritz_hist_cycles: Array


# ── main entry point ─────────────────────────────────────────────────


def arnoldi_iram_lm_jit(
    matvec,
    v0,
    *,
    m: int,
    nev: int = 1,
    k: int | None = None,
    max_cycles: int,
    tol: float,
    eps_breakdown: float = 0.0,
    stop_fn=None,
    compute_dtype=None,
):
    """
    Implicitly Restarted Arnoldi for the largest-magnitude eigenvalue(s).

    Parameters
    ----------
    matvec     : v -> A v
    v0         : (N,) initial vector
    m          : Krylov subspace size  (like NCV in ARPACK).
    nev        : number of eigenvalues actually wanted (default 1).
    k          : number of Ritz pairs to *retain* at each restart.
                 Must satisfy  nev <= k <= m - 1.
                 Default: min(max(2*nev, nev + 2), m - 1).
                 Using k > nev provides "guard" vectors that make the
                 implicit polynomial filter much more effective.
    max_cycles : max implicit restart cycles.
    tol        : convergence tolerance on the Ritz residual (or stop_fn).
    eps_breakdown : breakdown threshold.
    stop_fn    : optional  vec -> scalar  custom convergence metric.
    compute_dtype : working dtype.

    Returns
    -------
    theta : dominant eigenvalue approximation
    vec   : corresponding eigenvector (unit norm)
    info  : ArnoldiInfoJit

    Notes
    -----
    The effective number of matvecs per restart cycle is p = m - k
    (the extension phase).  More guard vectors (larger k) means fewer
    shifts and fewer new matvecs per cycle, but each cycle does less
    polynomial filtering.  The sweet spot is typically k ≈ 2 * nev.
    """
    # ── validate / default k ──────────────────────────────────────
    if k is None:
        k = min(max(2 * nev, nev + 2), m - 1)
    if not (nev <= k <= m - 1):
        raise ValueError(f'Need nev <= k <= m-1, got nev={nev}, k={k}, m={m}')
    if m < k + 1:
        raise ValueError(f'm must be >= k+1, got m={m}, k={k}')
    p = m - k  # number of shifts per cycle

    if compute_dtype is None:
        compute_dtype = v0.dtype
    v0 = jnp.asarray(v0, dtype=compute_dtype)
    dtype = v0.dtype
    real_dtype = jnp.real(v0).dtype
    int_dtype = jnp.int32

    eps_r = _as(eps_breakdown, real_dtype)
    tol_r = _as(tol, real_dtype)
    inf_r = _as(jnp.inf, real_dtype)
    zero_c = _as(0.0, dtype)
    N = v0.size

    # ── Arnoldi extension (j_start → m) ──────────────────────────
    def arnoldi_extend(V, H, j_start):
        js = jnp.arange(m, dtype=int_dtype)

        def step(carry, j):
            V, H = carry
            active = j >= j_start

            vj = V[j]
            w = jnp.asarray(matvec(vj), dtype=dtype)

            idx = jnp.arange(m + 1, dtype=int_dtype)
            mask = (idx <= j).astype(dtype)[:, None]
            Vm = V * mask

            # CGS2 (classical Gram-Schmidt with re-orthogonalisation)
            h = jnp.einsum('iN,N->i', jnp.conj(Vm), w)
            w = w - jnp.einsum('i,iN->N', h, Vm)
            h2 = jnp.einsum('iN,N->i', jnp.conj(Vm), w)
            w = w - jnp.einsum('i,iN->N', h2, Vm)
            h = h + h2

            h_next = jnp.linalg.norm(w)
            breakdown = h_next <= eps_r
            denom = jnp.where(breakdown, _as(jnp.inf, real_dtype), h_next)
            v_next = (w / denom).astype(dtype)

            new_col = h.at[j + 1].set(h_next.astype(dtype))

            col = jnp.where(active, new_col, H[:, j])
            v_row = jnp.where(active, v_next, V[j + 1])

            H = H.at[:, j].set(col)
            V = V.at[j + 1].set(v_row)
            return (V, H), None

        (V, H), _ = jax.lax.scan(step, (V, H), js)
        return V, H

    # ── initial state ────────────────────────────────────────────
    v_init = v0 / _safe_norm(v0, eps_r)
    V0 = jnp.zeros((m + 1, N), dtype=dtype).at[0].set(v_init)
    H0 = jnp.zeros((m + 1, m), dtype=dtype)

    metric_hist = jnp.full((max_cycles,), inf_r, dtype=real_dtype)
    ritz_hist = jnp.full((max_cycles,), inf_r, dtype=real_dtype)

    # ── while-loop ───────────────────────────────────────────────
    def cond(carry):
        cyc, _, _, _, _, _, best_metric, _, _, _ = carry
        return jnp.logical_and(cyc < max_cycles, best_metric > tol_r)

    def body(carry):
        (
            cyc,
            V,
            H,
            j_start,
            best_theta,
            best_vec,
            best_metric,
            best_ritz,
            metric_hist,
            ritz_hist,
        ) = carry

        # 1. Extend Arnoldi factorisation to full size m
        V, H = arnoldi_extend(V, H, j_start)

        # 2. Eigen-decompose the m × m Hessenberg
        Hm = H[:m, :m]
        evals, evecs = jnp.linalg.eig(Hm)

        # 3. Pick the dominant Ritz pair (largest magnitude)
        idx_best = jnp.argmax(jnp.abs(evals))
        theta = evals[idx_best]
        y = evecs[:, idx_best]

        # Ritz vector in the original space
        vec = V[:m].T @ y
        vec = vec / _safe_norm(vec, eps_r)

        # Ritz residual bound:  |h_{m+1,m}| · |e_m^T y|
        ritz_resid = jnp.abs(H[m, m - 1] * y[m - 1]).astype(real_dtype)
        metric = stop_fn(vec) if stop_fn is not None else ritz_resid
        metric = jnp.asarray(metric, dtype=real_dtype)

        metric_hist = metric_hist.at[cyc].set(metric)
        ritz_hist = ritz_hist.at[cyc].set(ritz_resid)

        improve = metric < best_metric
        best_metric = jnp.where(improve, metric, best_metric)
        best_theta = jnp.where(improve, theta, best_theta)
        best_ritz = jnp.where(improve, ritz_resid, best_ritz)
        best_vec = jax.lax.select(improve, vec, best_vec)

        # 4. Select p = m - k unwanted shifts (smallest-magnitude Ritz values)
        sort_idx = jnp.argsort(-jnp.abs(evals))  # descending magnitude
        shifts = evals[sort_idx[k:]]  # p smallest = unwanted

        # 5. Apply p implicit QR shifts to compress to k-step factorisation
        f_vec = H[m, m - 1] * V[m]
        Hs, Vs, f_new = _apply_shifts(Hm, V[:m], f_vec, m, k, shifts)

        # 6. Rebuild the compressed k-step factorisation for next cycle
        f_norm = _safe_norm(f_new, eps_r)
        v_kp1 = f_new / f_norm

        V_next = jnp.zeros((m + 1, N), dtype=dtype)
        H_next = jnp.zeros((m + 1, m), dtype=dtype)
        V_next = V_next.at[:k].set(Vs[:k])
        V_next = V_next.at[k].set(v_kp1)
        H_next = H_next.at[:k, :k].set(Hs[:k, :k])
        H_next = H_next.at[k, k - 1].set(f_norm.astype(dtype))

        return (
            cyc + 1,
            V_next,
            H_next,
            _as(k, int_dtype),
            best_theta,
            best_vec,
            best_metric,
            best_ritz,
            metric_hist,
            ritz_hist,
        )

    carry0 = (
        _as(0, int_dtype),
        V0,
        H0,
        _as(0, int_dtype),
        zero_c,
        v_init,
        inf_r,
        inf_r,
        metric_hist,
        ritz_hist,
    )

    result = jax.lax.while_loop(cond, body, carry0)
    cyc = result[0]
    best_theta = result[4]
    best_vec = result[5]
    best_ritz = result[7]
    metric_hist = result[8]
    ritz_hist = result[9]

    # n_inner_iters counts actual *useful* matvecs:
    #   first cycle does m, subsequent cycles do p = m - k each.
    #   total = m + (cyc - 1) * p   (for cyc >= 1)
    n_matvecs = m + (cyc - 1) * _as(p, int_dtype)
    n_matvecs = jnp.where(cyc > 0, n_matvecs, _as(0, int_dtype))

    info = ArnoldiInfoJit(
        eigval=best_theta,
        ritz_residual=best_ritz,
        n_cycles=cyc,
        n_inner_iters=n_matvecs,
        metric_hist_cycles=metric_hist,
        ritz_hist_cycles=ritz_hist,
    )
    return best_theta, best_vec, info
