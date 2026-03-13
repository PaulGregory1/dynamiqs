from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array


def _as(x, dtype):
    return jnp.asarray(x, dtype=dtype)


def _safe_norm(v: Array, eps: Array) -> Array:
    nrm = jnp.linalg.norm(v)
    return jnp.maximum(nrm, eps)


@dataclass
class ArnoldiInfoJit:
    eigval: Array
    ritz_residual: Array
    n_cycles: Array
    n_inner_iters: Array
    metric_hist_cycles: Array
    ritz_hist_cycles: Array


def arnoldi_restarted_lm_jit(
    matvec,
    v0,
    *,
    m: int,
    max_cycles: int,
    tol: float,
    eps_breakdown: float = 0.0,
    stop_fn=None,
    project_fn=None,
    compute_dtype=None,
):
    if m < 2:
        raise ValueError('m must be >= 2')

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
    v = v0 / _safe_norm(v0, eps_r)

    def one_cycle(v_start: Array):
        V0 = jnp.zeros((m + 1, N), dtype=dtype)
        H0 = jnp.zeros((m + 1, m), dtype=dtype)

        beta = _safe_norm(v_start, eps_r)
        v1 = v_start / beta
        V0 = V0.at[0].set(v1)

        js = jnp.arange(m, dtype=int_dtype)

        def step(carry, j):
            V, H = carry
            j = jnp.asarray(j, dtype=int_dtype)

            vj = V[j]
            w = jnp.asarray(matvec(vj), dtype=dtype)

            idx = jnp.arange(m + 1, dtype=int_dtype)
            mask = (idx <= j).astype(dtype)[:, None]
            Vmask = V * mask

            # --- Pass 1: Classical Gram-Schmidt ---
            h = jnp.einsum('iN,N->i', jnp.conj(Vmask), w)
            w = w - jnp.einsum('i,iN->N', h, Vmask)

            # --- Pass 2: re-orthogonalisation (CGS2) ---
            h2 = jnp.einsum('iN,N->i', jnp.conj(Vmask), w)
            w = w - jnp.einsum('i,iN->N', h2, Vmask)
            h = h + h2  # accumulate correction into h

            h_next = jnp.linalg.norm(w)
            H = H.at[:, j].set(h)
            H = H.at[j + 1, j].set(h_next.astype(dtype))

            breakdown = h_next <= eps_r
            denom = jnp.where(breakdown, jnp.inf, h_next)
            v_next = (w / denom).astype(dtype)

            V = V.at[j + 1].set(v_next)
            return (V, H), None

        (V, H), _ = jax.lax.scan(step, (V0, H0), js)

        Hm = H[:m, :m]
        evals, evecs = jnp.linalg.eig(Hm)
        idx = jnp.argmax(jnp.abs(evals))
        theta = evals[idx]
        y = evecs[:, idx]

        vec = V[:m].T @ y
        vec = vec / _safe_norm(vec, eps_r)

        ritz_resid = jnp.abs(H[m, m - 1] * y[m - 1]).astype(real_dtype)
        metric = stop_fn(vec) if stop_fn is not None else ritz_resid
        metric = jnp.asarray(metric, dtype=real_dtype)

        return theta, vec, metric, ritz_resid

    metric_hist = jnp.full((max_cycles,), inf_r, dtype=real_dtype)
    ritz_hist = jnp.full((max_cycles,), inf_r, dtype=real_dtype)

    best_theta = zero_c
    best_vec = v
    best_metric = inf_r
    best_ritz = inf_r

    def cond(carry):
        cyc, _, _, _, best_metric, _, _, _ = carry
        return jnp.logical_and(cyc < max_cycles, best_metric > tol_r)

    def body(carry):
        (
            cyc,
            v,
            best_theta,
            best_vec,
            best_metric,
            best_ritz,
            metric_hist,
            ritz_hist,
        ) = carry

        theta, vec, metric, ritz_resid = one_cycle(v)

        metric_hist = metric_hist.at[cyc].set(metric)
        ritz_hist = ritz_hist.at[cyc].set(ritz_resid)

        improve = metric < best_metric
        best_metric = jnp.where(improve, metric, best_metric)
        best_theta = jnp.where(improve, theta, best_theta)
        best_ritz = jnp.where(improve, ritz_resid, best_ritz)
        best_vec = jax.lax.select(improve, vec, best_vec)

        # Project the Ritz vector before using it as the next restart vector.
        # This hermitises and renormalises the density matrix interpretation
        # of the vector, keeping the restart in the physical subspace.
        restart_vec = project_fn(vec) if project_fn is not None else vec

        return (
            cyc + 1,
            restart_vec,
            best_theta,
            best_vec,
            best_metric,
            best_ritz,
            metric_hist,
            ritz_hist,
        )

    cyc0 = jnp.array(0, dtype=int_dtype)
    carry0 = (
        cyc0,
        v,
        best_theta,
        best_vec,
        best_metric,
        best_ritz,
        metric_hist,
        ritz_hist,
    )

    (cyc, v, best_theta, best_vec, best_metric, best_ritz, metric_hist, ritz_hist) = (
        jax.lax.while_loop(cond, body, carry0)
    )

    info = ArnoldiInfoJit(
        eigval=best_theta,
        ritz_residual=best_ritz,
        n_cycles=cyc,
        n_inner_iters=cyc * jnp.array(m, dtype=int_dtype),
        metric_hist_cycles=metric_hist,
        ritz_hist_cycles=ritz_hist,
    )

    return best_theta, best_vec, info
