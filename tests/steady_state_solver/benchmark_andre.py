"""
Benchmark steady-state solvers: GMRES vs dense.
Pure dynamiqs/JAX — no catographer dependency. Runs on higgs via anb_compute.

Hamiltonian matches fit_steady_state.py (Kerr buffer resonator):
  H = -K/2 (b†)²b² - δ b†b + ε b† + ε* b
  L = [sqrt(κ) b]
"""

# %% Local setup
from __future__ import annotations

import anb_compute
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch


# %% Benchmark function (runs on worker, only uses dynamiqs + jax)


def run_benchmark(truncations, krylov_sizes, batch_sizes):
    import time
    import dynamiqs as dq
    import jax
    import jax.numpy as jnp

    dq.set_matmul_precision('highest')
    dq.set_precision('double')

    twopi = 2 * jnp.pi

    # Representative parameter values (from fit_steady_state.py initial guess)
    kappa = 14.0 * twopi  # rad MHz
    kerr = -1.0 * twopi  # rad MHz
    delta = 0.0 * twopi  # rad MHz (resonator detuning)
    drive_eps = 5.0  # effective drive amplitude after TF (rad MHz)

    results = {
        'devices': str(jax.devices()),
        'backend': jax.default_backend(),
        'single_point': [],
        'batched': [],
    }

    def time_fn(fn, *args, n_iter=3):
        """Time a JIT-compiled function. Returns dict with jit_ms and run_ms list."""
        # First call = JIT compilation + execution
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        jit_ms = (time.perf_counter() - t0) * 1e3

        # Subsequent calls = pure execution
        run_times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            run_times.append((time.perf_counter() - t0) * 1e3)

        return {'jit_ms': jit_ms, 'run_ms': run_times}

    # ------------------------------------------------------------------
    # Dense solver (direct Lindbladian inversion)
    # ------------------------------------------------------------------
    def steady_state_dense(H, jump_ops):
        identity = dq.eye_like(H, layout=dq.dense).to_jax().flatten()
        A = dq.slindbladian(H, jump_ops).to_jax() + jnp.outer(identity, identity)
        rho = jnp.linalg.solve(A, identity)
        return rho.reshape(H.shape, order='F')

    # ------------------------------------------------------------------
    # Build parametric solve: H(delta_d) -> rho
    # ------------------------------------------------------------------
    def make_solve_fn(N, solver):
        b_op = dq.destroy(N)
        bdag_op = dq.dag(b_op)
        n_op = bdag_op @ b_op
        n2_op = n_op @ n_op
        drive_op = 1j * drive_eps * bdag_op + (-1j * drive_eps) * b_op
        L = [jnp.sqrt(kappa) * b_op.to_jax()]

        H_static = (-kerr / 2 * n2_op + drive_op).to_jax()

        if solver == 'dense':

            def solve(delta_d):
                H = H_static - (delta + delta_d) * n_op.to_jax()
                return steady_state_dense(H, L)
        else:

            def solve(delta_d):
                H = H_static - (delta + delta_d) * n_op.to_jax()
                result = dq.steadystate(H, L, solver=solver)
                return result.rho.to_jax()

        return solve

    # ------------------------------------------------------------------
    # 1. Single-point benchmark
    # ------------------------------------------------------------------
    delta_d_test = jnp.array(0.0)

    for N in truncations:
        solvers = [('dense', 'dense')]
        for ks in krylov_sizes:
            solvers.append((f'gmres_ks{ks}', dq.SteadyStateGMRES(krylov_size=ks)))

        for solver_name, solver in solvers:
            try:
                solve_fn = make_solve_fn(N, solver)
                fwd = jax.jit(solve_fn)
                jac = jax.jit(jax.jacfwd(solve_fn))

                fwd_timing = time_fn(fwd, delta_d_test)
                jac_timing = time_fn(jac, delta_d_test)

                results['single_point'].append(
                    {
                        'trunc': N,
                        'solver': solver_name,
                        'fwd': fwd_timing,
                        'jac': jac_timing,
                    }
                )
            except Exception as e:
                results['single_point'].append(
                    {'trunc': N, 'solver': solver_name, 'error': str(e)}
                )

    # ------------------------------------------------------------------
    # 2. Batched benchmark: vmap vs lax.map
    # ------------------------------------------------------------------
    for n_batch in batch_sizes:
        delta_d_sweep = jnp.linspace(-20 * twopi, 20 * twopi, n_batch)

        for N in truncations:
            solvers_batched = [
                ('dense', 'dense'),
                ('gmres_ks64', dq.SteadyStateGMRES(krylov_size=64)),
            ]

            for solver_name, solver in solvers_batched:
                solve_fn = make_solve_fn(N, solver)

                for batch_method in ['vmap', 'lax.map']:
                    try:
                        if batch_method == 'vmap':
                            batched = jax.jit(jax.vmap(solve_fn))
                        else:
                            batched = jax.jit(lambda x: jax.lax.map(solve_fn, x))

                        jac_batched = jax.jit(jax.jacfwd(batched))

                        fwd_timing = time_fn(batched, delta_d_sweep)
                        jac_timing = time_fn(jac_batched, delta_d_sweep)

                        results['batched'].append(
                            {
                                'n_batch': n_batch,
                                'trunc': N,
                                'solver': solver_name,
                                'method': batch_method,
                                'fwd': fwd_timing,
                                'jac': jac_timing,
                            }
                        )
                    except Exception as e:
                        results['batched'].append(
                            {
                                'n_batch': n_batch,
                                'trunc': N,
                                'solver': solver_name,
                                'method': batch_method,
                                'error': str(e),
                            }
                        )

    return results


# %% Submit to cluster
with patch('getpass.getuser', return_value='amelo'):
    with anb_compute.new_cluster(
        num_workers=1, worker_gpus=1, worker_memory=64, worker_cores=4, backend='higgs'
    ) as cluster:
        cluster.update_module('dynamiqs')

        future = cluster.client.submit(
            run_benchmark,
            truncations=[8, 16, 24, 32, 48],
            krylov_sizes=[32, 48, 64],
            batch_sizes=[5, 25],
        )
        results = future.result()


# %% Print results
def mean(lst):
    return sum(lst) / len(lst)


print(f'Devices: {results["devices"]}')
print(f'Backend: {results["backend"]}')

# --- Single-point table ---
print('\n## Single-point benchmark\n')
print(
    f'| {"N":>3} | {"solver":>12} | {"JIT fwd":>10} | {"JIT jac":>10} | {"fwd":>10} | {"jac":>10} | {"jac/fwd":>7} |'
)
print(f'|{"-" * 5}|{"-" * 14}|{"-" * 12}|{"-" * 12}|{"-" * 12}|{"-" * 12}|{"-" * 9}|')
for r in results['single_point']:
    if 'error' in r:
        print(
            f'| {r["trunc"]:>3} | {r["solver"]:>12} | {"ERROR":>10} | {r["error"][:30]}'
        )
    else:
        fwd_jit = r['fwd']['jit_ms']
        jac_jit = r['jac']['jit_ms']
        fwd_run = mean(r['fwd']['run_ms'])
        jac_run = mean(r['jac']['run_ms'])
        ratio = jac_run / fwd_run if fwd_run > 0 else float('inf')
        print(
            f'| {r["trunc"]:>3} | {r["solver"]:>12} '
            f'| {fwd_jit:>8.1f}ms | {jac_jit:>8.1f}ms '
            f'| {fwd_run:>8.2f}ms | {jac_run:>8.2f}ms '
            f'| {ratio:>5.1f}x |'
        )

# --- Batched table ---
for n_batch in sorted(set(r.get('n_batch', 0) for r in results['batched'])):
    batch_rows = [r for r in results['batched'] if r.get('n_batch') == n_batch]
    if not batch_rows:
        continue

    print(f'\n## Batched benchmark (batch={n_batch})\n')
    print(
        f'| {"N":>3} | {"solver":>12} | {"method":>8} | {"JIT fwd":>10} | {"JIT jac":>10} | {"fwd":>10} | {"jac":>10} | {"jac/fwd":>7} |'
    )
    print(
        f'|{"-" * 5}|{"-" * 14}|{"-" * 10}|{"-" * 12}|{"-" * 12}|{"-" * 12}|{"-" * 12}|{"-" * 9}|'
    )

    for r in batch_rows:
        if 'error' in r:
            print(
                f'| {r["trunc"]:>3} | {r["solver"]:>12} | {r["method"]:>8} | {"ERROR":>10} | {r["error"][:40]}'
            )
        else:
            fwd_jit = r['fwd']['jit_ms']
            jac_jit = r['jac']['jit_ms']
            fwd_run = mean(r['fwd']['run_ms'])
            jac_run = mean(r['jac']['run_ms'])
            ratio = jac_run / fwd_run if fwd_run > 0 else float('inf')
            print(
                f'| {r["trunc"]:>3} | {r["solver"]:>12} | {r["method"]:>8} '
                f'| {fwd_jit:>8.1f}ms | {jac_jit:>8.1f}ms '
                f'| {fwd_run:>8.2f}ms | {jac_run:>8.2f}ms '
                f'| {ratio:>5.1f}x |'
            )
