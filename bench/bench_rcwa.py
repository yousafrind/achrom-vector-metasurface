"""Throughput of the eig-free RCWA at realistic orders.  Usage: python bench/bench_rcwa.py [--cpu-ok]
Prints a table and writes bench/bench_rcwa.json.  Send the JSON back (or git push it)."""
import sys, time, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import jax, jax.numpy as jnp, numpy as np
from achrom.rcwa import _solve_cell, rect_eps, auto_k

dev = jax.devices()[0]
if dev.platform == "cpu" and "--cpu-ok" not in sys.argv:
    sys.exit(f"JAX is on CPU ({dev}); the ROCm plugin did not load. Pass --cpu-ok to run anyway.")
print("device:", dev)
LAM, P, H, B = 0.532, 0.4, 0.6, 8
res = {}
for M in (5, 7, 9, 11):
    k = auto_k(H, LAM, P, P, M, M)
    ws = jnp.linspace(0.13, 0.33, B)
    eps = jax.vmap(lambda w: rect_eps(w, 0.25, P, P, 2.4**2, Nx=64 + 4 * M, Ny=64 + 4 * M))(ws)
    f = jax.jit(jax.vmap(lambda e: _solve_cell(e, H, LAM, P, P, 1.46**2, 1.0, M, M, k)))
    try:
        t0 = time.perf_counter(); jax.block_until_ready(f(eps)); tc = time.perf_counter() - t0
        t0 = time.perf_counter(); out = f(eps); jax.block_until_ready(out); t = (time.perf_counter() - t0) / B
        e = float(jnp.max(jnp.abs(out["T"] + out["R"] - 1)))
        res[M] = dict(orders=(2 * M + 1) ** 2, k=k, sec_per_cell=t, compile_s=tc, energy_err=e)
    except Exception as ex:                                   # report, keep going
        res[M] = dict(error=str(ex)[:200])
    print(M, res[M], flush=True)
json.dump(res, open(pathlib.Path(__file__).with_name("bench_rcwa.json"), "w"), indent=2)
