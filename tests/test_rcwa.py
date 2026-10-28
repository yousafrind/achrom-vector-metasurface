"""Validation of the eig-free RCWA.  Run: python tests/test_rcwa.py  (or pytest)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import jax, jax.numpy as jnp
from achrom.rcwa import solve as _solve, rect_eps, auto_k


def solve(eps, h, lam, px, py, ei, eo, Mx=5, My=5, k=None):
    k = auto_k(h, lam, px, py, Mx, My) if k is None else k
    return _solve(eps, h, lam, px, py, ei, eo, Mx=Mx, My=My, k=k)

LAM = 0.532


def film(n_film, n_in, n_out, d, lam=LAM):
    """Exact E-amplitude transmission of a uniform film, power-normalised."""
    t01, t12 = 2 * n_in / (n_in + n_film), 2 * n_film / (n_film + n_out)
    r01, r12 = (n_in - n_film) / (n_in + n_film), (n_film - n_out) / (n_film + n_out)
    ph = 2 * np.pi * n_film * d / lam
    return np.sqrt(n_out / n_in) * t01 * t12 * np.exp(1j * ph) / (1 + r01 * r12 * np.exp(2j * ph))


def test_uniform_film_vs_fresnel():
    for n_in, n_f, n_out, d in [(1.0, 2.4, 1.0, 0.3), (1.46, 2.4, 1.0, 0.6), (1.0, 1.5, 1.46, 0.05)]:
        eps = jnp.full((8, 8), n_f**2)
        out = solve(eps, d, LAM, 0.4, 0.4, n_in**2, n_out**2, Mx=2, My=2)
        ref = film(n_f, n_in, n_out, d)
        assert abs(out["J"][0, 0] - ref) < 1e-9, (n_in, n_f, n_out, abs(out["J"][0, 0] - ref))
        assert abs(out["J"][1, 1] - ref) < 1e-9
        assert abs(out["J"][0, 1]) < 1e-12
        assert abs(out["T"][0] + out["R"][0] - 1) < 1e-9


def _rytov_neff(p, f, e1, e2, tm, lam=LAM):
    """Exact fundamental-mode effective index of a lamellar grating (Rytov dispersion relation)."""
    from scipy.optimize import brentq
    k0 = 2 * np.pi / lam

    def F(b):
        k1, k2 = k0 * np.sqrt(e1 - b * b + 0j), k0 * np.sqrt(e2 - b * b + 0j)
        c = (e2 * k1 / (e1 * k2) + e1 * k2 / (e2 * k1)) if tm else (k1 / k2 + k2 / k1)
        return (np.cos(k1 * f * p) * np.cos(k2 * (1 - f) * p) - 0.5 * c * np.sin(k1 * f * p) * np.sin(k2 * (1 - f) * p) - 1).real
    n0 = 1 / np.sqrt(f / e1 + (1 - f) / e2) if tm else np.sqrt(f * e1 + (1 - f) * e2)
    return brentq(F, n0 - 0.3, n0 + 0.3)


def test_effective_medium_lamellar():
    """Subwavelength lamellar grating (lines along y): E||lines = TE, E perp = TM.
    TE must match a film with the exact Rytov index to 1e-3 (error ~ p^2).  TM keeps a
    first-order-in-p interface residual (0.0079 at p=lam/20, halving with p) -- checked to 0.015."""
    f, e1, p, d = 0.5, 4.0, LAM / 20, 0.3
    x = (np.arange(256) + 0.5) / 256
    eps = jnp.asarray(np.where(x < f, e1, 1.0))[None, :]
    out = solve(eps, d, LAM, p, p, 1.0, 1.0, Mx=15, My=0)
    te, tm = _rytov_neff(p, f, e1, 1.0, False), _rytov_neff(p, f, e1, 1.0, True)
    assert abs(out["J"][1, 1] - film(te, 1, 1, d)) < 1e-3
    assert abs(out["J"][0, 0] - film(tm, 1, 1, d)) < 0.015


def test_inverse_rule_matches_laurent_limit():
    """Li's inverse rule (fast) and naive Laurent (slow, ~1/M) must converge to the same TM answer."""
    import achrom.rcwa as R
    f, e1, d, p = 0.5, 4.0, 0.3, 0.3
    x = (np.arange(1024) + 0.5) / 1024
    eps = jnp.asarray(np.where(x < f, e1, 1.0))[None, :]
    li = solve(eps, d, LAM, p, p, 1.0, 1.0, Mx=20, My=0)["J"][0, 0]
    jax.clear_caches()
    orig, R._eps_x = R._eps_x, (lambda e, Mx, My: R._laurent(e, Mx, My))
    try:
        dev = [abs(solve(eps, d, LAM, p, p, 1.0, 1.0, Mx=M, My=0)["J"][0, 0] - li) for M in (60, 120)]
    finally:
        R._eps_x = orig
        jax.clear_caches()
    print("Laurent deviation from Li at M=60,120:", [round(float(v), 5) for v in dev])
    assert dev[1] < dev[0] and dev[1] < 3e-3


def test_energy_and_symmetry_pillar():
    eps = rect_eps(0.16, 0.28, 0.4, 0.4, 2.4**2)
    out = solve(eps, 0.6, LAM, 0.4, 0.4, 1.46**2, 1.0, Mx=7, My=7)
    assert np.all(np.abs(np.asarray(out["T"] + out["R"]) - 1) < 1e-7), (out["T"], out["R"])
    assert abs(out["J"][0, 1]) < 1e-9 and abs(out["J"][1, 0]) < 1e-9          # mirror symmetry
    sw = solve(rect_eps(0.28, 0.16, 0.4, 0.4, 2.4**2), 0.6, LAM, 0.4, 0.4, 1.46**2, 1.0, Mx=7, My=7)
    assert abs(sw["J"][1, 1] - out["J"][0, 0]) < 1e-9                              # x<->y swap
    sq = solve(rect_eps(0.2, 0.2, 0.4, 0.4, 2.4**2), 0.6, LAM, 0.4, 0.4, 1.46**2, 1.0, Mx=7, My=7)
    assert abs(sq["J"][0, 0] - sq["J"][1, 1]) < 1e-9                                # C4


def test_thick_layer_needs_doubling():
    """k=0 (single expm, no doubling) vs k=4 on a thick high-order layer: energy conservation."""
    eps = rect_eps(0.16, 0.28, 0.4, 0.4, 2.4**2)
    err = {}
    for k in (0, 4):
        o = solve(eps, 1.2, LAM, 0.4, 0.4, 1.46**2, 1.0, Mx=7, My=7, k=k)
        err[k] = float(np.max(np.abs(np.asarray(o["T"] + o["R"]) - 1)))
    print("energy error k=0: %.2e  k=4: %.2e" % (err[0], err[4]))
    assert err[4] < 1e-7


def test_gradients_vs_finite_difference():
    def f(h, w):
        o = solve(rect_eps(w, 0.28, 0.4, 0.4, 2.4**2), h, LAM, 0.4, 0.4, 1.46**2, 1.0, Mx=4, My=4, k=3)
        return jnp.abs(o["J"][0, 0])**2 + jnp.angle(o["J"][1, 1])
    g = jax.grad(f, argnums=(0, 1))(0.6, 0.171)
    for i, (h, w) in enumerate([(1e-5, 0.0), (0.0, 1e-5)]):
        fd = (f(0.6 + h, 0.171 + w) - f(0.6 - h, 0.171 - w)) / 2e-5
        assert abs(g[i] - fd) < 1e-5 * max(1, abs(fd)), (i, g[i], fd)


def test_convergence_in_orders():
    eps = rect_eps(0.16, 0.28, 0.4, 0.4, 2.4**2)
    J = {M: solve(eps, 0.6, LAM, 0.4, 0.4, 1.46**2, 1.0, Mx=M, My=M)["J"][0, 0] for M in (5, 7, 9)}
    print("convergence |J_xx|:", {M: round(float(abs(v)), 5) for M, v in J.items()},
          " phase:", {M: round(float(np.angle(v)), 4) for M, v in J.items()})
    assert abs(J[9] - J[7]) < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
