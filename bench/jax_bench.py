"""Hardware check for JAX-based metasurface inverse design (RCWA + FFT propagation).

Usage:  python jax_bench.py            # full sizes (run this on the Ryzen AI Max)
        python jax_bench.py --small    # tiny sizes, for a quick sanity run

Each stage runs in its own subprocess with a timeout, so a GPU hang shows up as
'timeout' instead of freezing the whole run. Results go to bench_results.json.
Send that file (and the printed table) back.
"""
import json, subprocess, sys, time

SMALL = "--small" in sys.argv
N_FFT = 1024 if SMALL else 4096          # field grid side
N_FFT_BIG = 2048 if SMALL else 8192
N_MAT = 60 if SMALL else 242             # RCWA-like matrix size, 2*(2M+1)^2 with M=5
N_BATCH = 8 if SMALL else 256            # unit cells per batch
TIMEOUT = 120 if SMALL else 600

STAGES = ["devices", "bandwidth", "fft_c64", "fft_c64_big", "fft_c128",
          "eig", "expm", "solve", "grad_propagation"]


def timed(fn, *a, reps=3):
    import jax
    out = fn(*a); jax.block_until_ready(out)            # compile + warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*a); jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps


def run_stage(name):
    import jax, jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    if name == "devices":
        return {"devices": [str(d) for d in jax.devices()], "jax": jax.__version__}
    if name == "bandwidth":                              # elementwise y = 2x + 1, ~read+write
        x = jnp.ones((N_FFT_BIG * N_FFT_BIG // 4,), jnp.float32)
        t = timed(jax.jit(lambda v: 2 * v + 1), x)
        return {"GB_per_s": 2 * x.nbytes / t / 1e9}
    if name in ("fft_c64", "fft_c64_big"):
        n = N_FFT if name == "fft_c64" else N_FFT_BIG
        x = jnp.ones((n, n), jnp.complex64)
        t = timed(jax.jit(lambda v: jnp.fft.ifft2(jnp.fft.fft2(v))), x)
        return {"n": n, "sec_fft_ifft_pair": t}
    if name == "fft_c128":
        jax.config.update("jax_enable_x64", True)
        x = jnp.ones((N_FFT, N_FFT), jnp.complex128)
        t = timed(jax.jit(lambda v: jnp.fft.ifft2(jnp.fft.fft2(v))), x)
        return {"n": N_FFT, "sec_fft_ifft_pair": t}
    if name in ("eig", "expm", "solve"):
        jax.config.update("jax_enable_x64", True)        # RCWA is usually run in float64
        a = (jax.random.normal(key, (N_BATCH, N_MAT, N_MAT))
             + 1j * jax.random.normal(key, (N_BATCH, N_MAT, N_MAT))) / N_MAT ** 0.5
        if name == "eig":
            f = jax.jit(lambda m: jnp.linalg.eig(m))
        elif name == "expm":
            from jax.scipy.linalg import expm
            f = jax.jit(jax.vmap(expm))
        else:
            b = jnp.ones((N_BATCH, N_MAT, N_MAT), a.dtype)
            f = jax.jit(lambda m: jnp.linalg.solve(m + 2 * jnp.eye(N_MAT), b))
        t = timed(f, a, reps=2)
        out = f(a); jax.block_until_ready(out)
        leaf = jax.tree_util.tree_leaves(out)[0]
        return {"batch": N_BATCH, "size": N_MAT, "sec": t,
                "output_device": str(list(leaf.devices())[0])}
    if name == "grad_propagation":                       # 3 wavelengths, angular-spectrum + grad
        n = N_FFT
        phase = jax.random.uniform(key, (n, n), jnp.float32)
        fx = jnp.fft.fftfreq(n, d=0.3)                   # um^-1, 0.3 um pitch
        FX, FY = jnp.meshgrid(fx, fx)

        def loss(ph):
            tot = 0.0
            for lam in (0.450, 0.532, 0.635):
                field = jnp.exp(1j * ph * (0.532 / lam))
                kz2 = jnp.maximum(1 / lam ** 2 - FX ** 2 - FY ** 2, 0.0)
                H = jnp.exp(2j * jnp.pi * 200.0 * jnp.sqrt(kz2))
                out = jnp.fft.ifft2(jnp.fft.fft2(field) * H)
                tot = tot + jnp.mean(jnp.abs(out) ** 2 * (FX[0, 0] + 1))
            return tot
        t = timed(jax.jit(jax.grad(loss)), phase, reps=2)
        return {"n": n, "sec_per_grad_step_3wl": t}
    raise ValueError(name)


def main():
    if "--stage" in sys.argv:
        name = sys.argv[sys.argv.index("--stage") + 1]
        print("RESULT " + json.dumps(run_stage(name)))
        return
    results = {}
    for s in STAGES:
        t0 = time.time()
        cmd = [sys.executable, __file__, "--stage", s] + (["--small"] if SMALL else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if p.returncode == 0 and line:
                r = {"status": "ok", **json.loads(line[-1][7:])}
            else:
                err = (p.stderr.strip().splitlines() or ["no output"])[-1]
                r = {"status": "fail", "error": err[:300]}
        except subprocess.TimeoutExpired:
            r = {"status": "timeout"}
        r["wall_s"] = round(time.time() - t0, 1)
        results[s] = r
        print(f"{s:18s} {r['status']:8s} {json.dumps({k: v for k, v in r.items() if k != 'status'})}", flush=True)
    json.dump(results, open("bench_results.json", "w"), indent=2)
    print("\nWrote bench_results.json")


if __name__ == "__main__":
    main()
