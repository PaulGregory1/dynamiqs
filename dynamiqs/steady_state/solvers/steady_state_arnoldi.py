from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

import dynamiqs as dq

from ...options import Options
from ...qarrays.qarray import QArray
from ..api.steady_state_solver import SteadyStateResult, SteadyStateSolver
from ..api.utils import (
    finalize_density_matrix,
    from_dm,
    from_matrix,
    to_dm,
    to_matrix,
    update_preconditioner,
)
from ..core.arnoldi import arnoldi_restarted_lm_jit
from ..core.arnoldi_iram import arnoldi_iram_lm_jit
from ..preconditionner.lyapunov_solver import LyapunovSolverEig


class SteadyStateArnoldiResult(SteadyStateResult):
    """Result of the Arnoldi steady-state solver.

    Attributes:
        rho: The steady-state density matrix, of shape `(..., n, n)`.
    """

    rho: QArray

    @staticmethod
    def out_axes() -> SteadyStateArnoldiResult:
        return SteadyStateArnoldiResult(rho=0)


class SteadyStateArnoldi(SteadyStateSolver):
    r"""Arnoldi-based steady-state solver.

    Finds the steady state by computing the dominant eigenvector of the
    preconditioned Kraus map $T = S^{-1} \circ \mathcal{K}$, where
    $\mathcal{K}(\rho) = \rho + \mathcal{L}(\rho)$ is the Kraus map associated
    with the Lindbladian $\mathcal{L}$, and $S^{-1}$ is a Lyapunov-based
    preconditioner.

    The dominant eigenvector of $T$ (eigenvalue $\approx 1$) corresponds to the
    steady state $\rho_\infty$ such that $\mathcal{L}(\rho_\infty) = 0$.

    The eigenvector is found using implicitly restarted Arnoldi iteration.
    Differentiation is handled by `jax.lax.custom_linear_solve` applied to the
    deflated Lindbladian system, identical to the GMRES solver.

    The preconditioner is built from the Lyapunov part of the Lindbladian.
    Defining
    $$
        G = iH + \tfrac{1}{2}\sum_k L_k^\dagger L_k,
    $$
    the action of $S^{-1}$ on a matrix $Y$ is defined by solving the Lyapunov
    equation $G X + X G^\dagger = Y$.

    Attributes:
        tol: Tolerance for the stopping criterion. The solver stops when
            $\max|\mathcal{L}(\rho)| < \mathrm{tol}$. Defaults to `1e-6`.
        max_cycles: Maximum number of Arnoldi restart cycles. Defaults to `50`.
        krylov_size: Size of the Krylov subspace per Arnoldi cycle.
            Defaults to `30`. Must be >= 2.
        exact_dm: If `True`, project the final matrix onto the set of valid
            density matrices (positive semidefinite with unit trace). If `False`,
            only Hermitization and trace normalization are applied.
            Defaults to `True`.
        eps_breakdown: Threshold for detecting Arnoldi breakdown.
            Defaults to `0.0`.

    Examples:
        ```python
        import dynamiqs as dq

        n = 16
        a = dq.destroy(n)
        H = a + a.dag()
        jump_ops = [a]

        # Default parameters
        solver = dq.SteadyStateArnoldi()
        result = dq.steadystate(H, jump_ops, solver=solver)

        # Custom parameters
        solver = dq.SteadyStateArnoldi(tol=1e-8, krylov_size=40, max_cycles=100)
        result = dq.steadystate(H, jump_ops, solver=solver)

        print(result.rho)
        ```
    """

    tol: float = 1e-6
    max_cycles: int = 50
    krylov_size: int = 100
    exact_dm: bool = True
    eps_breakdown: float = 0.0
    n_refinement: int = 0

    @staticmethod
    def result_type() -> type[SteadyStateArnoldiResult]:
        return SteadyStateArnoldiResult

    def _run(
        self, H: QArray, Ls: list[QArray], rho0: QArray | None, options: Options
    ) -> SteadyStateArnoldiResult:
        del options

        n = H.shape[-1]
        dims = H.dims
        tol = self.tol
        max_cycles = self.max_cycles
        krylov_size = min(self.krylov_size, n * n - 1)

        H_jax = H.to_jax()
        Ls_jax = [L.to_jax() for L in Ls]
        dtype = H_jax.dtype

        identity_vec = from_matrix(jnp.eye(n, dtype=dtype))

        if rho0 is None:
            rho0 = dq.asqarray(jnp.eye(n, dtype=dtype) / n, dims=dims)
        x_0 = from_dm(rho0)

        H_q = dq.asqarray(H_jax, dims=dims)
        Ls_q = dq.stack(
            [dq.asqarray(L_j, dims=Ls[i].dims) for i, L_j in enumerate(Ls_jax)]
        )

        # ── Lindbladian and Kraus map ───────────────────────────────────────

        def lindbladian_vec(x: jax.Array) -> jax.Array:
            return from_dm(dq.lindbladian(H_q, Ls_q, to_dm(x, n=n, dims=dims)))

        def kraus_vec(x: jax.Array) -> jax.Array:
            """Kraus map: K(rho) = sum_k L_k rho L_k^dag."""
            rho = to_dm(x, n=n, dims=dims)
            return from_matrix((Ls_q @ rho @ Ls_q.dag()).sum(0).to_jax())

        def deflated_matvec(x: jax.Array) -> jax.Array:
            return lindbladian_vec(x) + identity_vec * jnp.dot(identity_vec, x)

        # ── Preconditioner (stop_gradient: not differentiated) ──────────────

        LdagL = (Ls_q.dag() @ Ls_q).sum(0).to_jax()
        G = jax.lax.stop_gradient(1j * H_jax + 0.5 * LdagL)

        def make_preconditioner(G_mat: jax.Array) -> Callable[[jax.Array], jax.Array]:
            solver = LyapunovSolverEig(G_mat, n_refinement=self.n_refinement)

            def precond(x: jax.Array) -> jax.Array:
                return from_matrix(solver.solve(to_matrix(x, n=n), mu=0.0))

            return precond

        precond_fn = make_preconditioner(G)
        precond_fn_adj = make_preconditioner(G.conj().T)

        # ── Arnoldi: find dominant eigenvector of T = precond @ kraus ───────

        # ── Arnoldi: find dominant eigenvector of T = precond @ kraus ───────
        # We stop gradients on ALL inputs to Arnoldi so JAX never tries to
        # differentiate through jnp.linalg.eig inside the Arnoldi iteration.

        H_jax_sg = jax.lax.stop_gradient(H_jax)
        Ls_jax_sg = [jax.lax.stop_gradient(L_j) for L_j in Ls_jax]

        H_q_sg = dq.asqarray(H_jax_sg, dims=dims)
        Ls_q_sg = dq.stack(
            [dq.asqarray(L_j, dims=Ls[i].dims) for i, L_j in enumerate(Ls_jax_sg)]
        )

        def lindbladian_vec_sg(x: jax.Array) -> jax.Array:
            return from_dm(dq.lindbladian(H_q_sg, Ls_q_sg, to_dm(x, n=n, dims=dims)))

        def kraus_vec_sg(x: jax.Array) -> jax.Array:
            rho = to_dm(x, n=n, dims=dims)
            return from_matrix((Ls_q_sg @ rho @ Ls_q_sg.dag()).sum(0).to_jax())

        LdagL_sg = (Ls_q_sg.dag() @ Ls_q_sg).sum(0).to_jax()
        G_sg = 1j * H_jax_sg + 0.5 * LdagL_sg
        precond_fn_sg = make_preconditioner(G_sg)

        def precond_kraus_vec(x: jax.Array) -> jax.Array:
            return kraus_vec_sg(precond_fn_sg(x))

        def stop_metric_from_vec(vec: jax.Array) -> jax.Array:
            rho = to_matrix(vec, n=n)
            rho = 0.5 * (rho + rho.conj().mT)
            tr = jnp.trace(rho)
            tr = jnp.where(jnp.abs(tr) > 0, tr, jnp.array(1.0, dtype=rho.dtype))
            rho = rho / tr
            rho_q = dq.asqarray(rho, dims=dims)
            L_rho = dq.lindbladian(H_q_sg, Ls_q_sg, rho_q).to_jax()
            return jnp.max(jnp.abs(L_rho))

        v0 = jax.lax.stop_gradient(x_0)

        def project_fn(vec):
            """Hermitise + renormalise la matrice densité vectorisée."""
            rho = to_matrix(vec, n=n)
            rho = 0.5 * (rho + rho.conj().mT)
            tr = jnp.trace(rho)
            tr = jnp.where(jnp.abs(tr) > 0, tr, jnp.array(1.0, dtype=rho.dtype))
            rho = rho / tr
            return from_matrix(rho)

        # _eigval, eigvec, _info = arnoldi_restarted_lm_jit(
        #     precond_kraus_vec,
        #     v0,
        #     m=krylov_size,
        #     max_cycles=max_cycles,
        #     tol=tol,
        #     eps_breakdown=self.eps_breakdown,
        #     stop_fn=stop_metric_from_vec,
        #     project_fn=project_fn,
        # )

        _eigval, eigvec, _info = arnoldi_iram_lm_jit(
            precond_kraus_vec,
            v0,
            m=krylov_size,
            nev=1,
            k=krylov_size - 1,
            max_cycles=max_cycles,
            tol=tol,
            eps_breakdown=self.eps_breakdown,
            stop_fn=stop_metric_from_vec,
        )

        # ── Build forward solution from Arnoldi eigenvector ─────────────────
        rho_arnoldi = to_matrix(precond_fn_sg(eigvec), n=n)
        rho_arnoldi = 0.5 * (rho_arnoldi + rho_arnoldi.conj().mT)
        tr = jnp.trace(rho_arnoldi)
        tr = jnp.where(jnp.abs(tr) > 0, tr, jnp.array(1.0, dtype=rho_arnoldi.dtype))
        rho_arnoldi = rho_arnoldi / tr
        x_arnoldi = from_matrix(rho_arnoldi)

        # ── Differentiable wrapper via custom_linear_solve ──────────────────
        # Arnoldi gives the forward solution but is not differentiable.
        # custom_linear_solve defines the implicit differentiation rule:
        # rho satisfies (L + |I><I|)|x> = |I>, so gradients use the adjoint.

        # def solve(matvec: Callable[[jax.Array], jax.Array], b: jax.Array) -> jax.Array:
        #     del matvec, b
        #     return x_arnoldi  # already stop_gradient'd through inputs

        def solve(matvec, b):
            def right_precond_matvec(y):
                return matvec(precond_fn(y))

            z, _info = jax.scipy.sparse.linalg.gmres(
                right_precond_matvec,
                b,
                x0=x_arnoldi,  # warm start from Arnoldi
                tol=tol,
                restart=krylov_size,
                maxiter=100,
            )
            return precond_fn(z)

        def transpose_solve(
            matvec_adj: Callable[[jax.Array], jax.Array], b: jax.Array
        ) -> jax.Array:
            def right_matvec_adj(y: jax.Array) -> jax.Array:
                return precond_fn_adj(matvec_adj(y))

            y, _info = jax.scipy.sparse.linalg.gmres(
                right_matvec_adj,
                b,
                x0=jnp.zeros_like(b),
                tol=tol,
                restart=krylov_size,
                maxiter=100,
            )
            return precond_fn_adj(y)

        x_sol = jax.lax.custom_linear_solve(
            deflated_matvec,
            identity_vec,
            solve=solve,
            transpose_solve=transpose_solve,
            symmetric=False,
        )

        rho_ss = finalize_density_matrix(to_matrix(x_sol, n=n), self.exact_dm)
        return SteadyStateArnoldiResult(rho=dq.asqarray(rho_ss, dims=dims))
