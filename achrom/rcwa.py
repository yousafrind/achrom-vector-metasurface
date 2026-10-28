"""Eig-free RCWA in JAX: normal incidence, 2D-periodic, one patterned layer.

Conventions: time dependence exp(-i w t); lengths in the same unit as `lam`;
H is normalised as eta0*H; field vector per layer = [Ex; Ey; Hx; Hy] over Fourier orders
(order index n-major: idx = n*nx + m).  Fourier factorisation: Li's inverse rule along the
field's normal direction for Ex, Ey (per pixel row / column), Laurent rule for Ez.

Layer propagator exp(i k0 d G) is built with expm on a thin slab (d / 2**k) and doubled in
scattering-matrix form (Redheffer star), which avoids the growing exponentials of a direct
transfer matrix.  No eigendecomposition anywhere (gradient-safe at degenerate modes).
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------- Fourier machinery
def _orders(M):
    return jnp.arange(-M, M + 1)


def _diff_idx(M, N):
    r = _orders(M)
    return (r[:, None] - r[None, :]) % N


def _laurent(f, Mx, My):
    """Toeplitz-Toeplitz convolution matrix of f(Ny, Nx), shape (N, N), N=nx*ny."""
    Ny, Nx = f.shape
    nx, ny = 2 * Mx + 1, 2 * My + 1
    F = jnp.fft.fft2(f) / (Nx * Ny)
    C = F[_diff_idx(My, Ny)[:, None, :, None], _diff_idx(Mx, Nx)[None, :, None, :]]
    return C.reshape(nx * ny, nx * ny)


def _eps_x(eps, Mx, My):
    """Operator Ex -> Dx: inverse rule along x (per row), Laurent along y."""
    Ny, Nx = eps.shape
    nx, ny = 2 * Mx + 1, 2 * My + 1
    G = jnp.fft.fft(1.0 / eps, axis=1) / Nx
    A = jnp.linalg.inv(G[:, _diff_idx(Mx, Nx)])                 # (Ny, nx, nx)
    Ah = jnp.fft.fft(A, axis=0) / Ny
    return jnp.transpose(Ah[_diff_idx(My, Ny)], (0, 2, 1, 3)).reshape(nx * ny, nx * ny)


def _eps_y(eps, Mx, My):
    """Operator Ey -> Dy: inverse rule along y (per column), Laurent along x."""
    Ny, Nx = eps.shape
    nx, ny = 2 * Mx + 1, 2 * My + 1
    G = jnp.fft.fft(1.0 / eps, axis=0) / Ny
    A = jnp.linalg.inv(jnp.moveaxis(G[_diff_idx(My, Ny)], 2, 0))  # (Nx, ny, ny)
    Ah = jnp.fft.fft(A, axis=0) / Nx
    return jnp.transpose(Ah[_diff_idx(Mx, Nx)], (2, 0, 3, 1)).reshape(nx * ny, nx * ny)


def _K(lam, px, py, Mx, My):
    m = jnp.tile(_orders(Mx), 2 * My + 1)
    n = jnp.repeat(_orders(My), 2 * Mx + 1)
    return m * lam / px, n * lam / py


# ---------------------------------------------------------------- modes and S-matrices
def _kz(eps, Kx, Ky):
    kz = jnp.sqrt(eps - Kx**2 - Ky**2 + 0j)
    kz = jnp.where(kz.imag < 0, -kz, kz)
    return kz + 1e-12j                                          # keep off Wood anomalies


def _medium(eps, Kx, Ky):
    """Forward/backward mode matrix W=[[I,I],[V,-V]], its inverse, and kz (per order)."""
    N = Kx.size
    kz = _kz(eps, Kx, Ky)
    d = lambda v: jnp.diag(v)
    Q = jnp.block([[-d(Kx * Ky), d(Kx**2) - eps * jnp.eye(N)],
                   [eps * jnp.eye(N) - d(Ky**2), d(Ky * Kx)]])
    kz2 = jnp.concatenate([kz, kz])
    V = Q / kz2[None, :]
    Vi = kz2[:, None] * jnp.linalg.inv(Q)
    I = jnp.eye(2 * N)
    W = jnp.block([[I, I], [V, -V]])
    Wi = 0.5 * jnp.block([[I, Vi], [I, -Vi]])
    return W, Wi, kz, V


def _T2S(T):
    n = T.shape[0] // 2
    T11, T12, T21, T22 = T[:n, :n], T[:n, n:], T[n:, :n], T[n:, n:]
    T22i = jnp.linalg.inv(T22)
    return (-T22i @ T21, T22i, T11 - T12 @ T22i @ T21, T12 @ T22i)   # R, T', T, R'


def _star(A, B):
    RA, TpA, TA, RpA = A
    RB, TpB, TB, RpB = B
    I = jnp.eye(RA.shape[0], dtype=RA.dtype)
    X = jnp.linalg.inv(I - RpA @ RB)
    Y = jnp.linalg.inv(I - RB @ RpA)
    return (RA + TpA @ RB @ X @ TA, TpA @ Y @ TpB, TB @ X @ TA, RpB + TB @ RpA @ Y @ TpB)


def _layer_P_Q(eps, Kx, Ky, Mx, My):
    N = Kx.size
    Ei = jnp.linalg.inv(_laurent(eps, Mx, My))
    ex, ey = _eps_x(eps, Mx, My), _eps_y(eps, Mx, My)
    d = jnp.diag
    I = jnp.eye(N)
    KxEi, KyEi = Kx[:, None] * Ei, Ky[:, None] * Ei
    P = jnp.block([[KxEi * Ky[None, :], I - KxEi * Kx[None, :]],
                   [KyEi * Ky[None, :] - I, -KyEi * Kx[None, :]]])
    Q = jnp.block([[-d(Kx * Ky), d(Kx**2) - ey], [ex - d(Ky**2), d(Ky * Kx)]])
    return P, Q


def auto_k(h, lam, px, py, Mx, My, target=6.0):
    """Doublings so the thin-slab exponent kappa_max*k0*h/2**k stays <= target (needs concrete h)."""
    kmax = ((Mx * lam / px) ** 2 + (My * lam / py) ** 2) ** 0.5
    x = kmax * 2 * float(jnp.pi) * float(h) / lam / target
    return max(0, int(jnp.ceil(jnp.log2(x)))) if x > 1 else 0


@partial(jax.jit, static_argnames=("Mx", "My", "k"))
def solve(eps, h, lam, px, py, eps_in, eps_out, Mx=5, My=5, k=3):
    """Zeroth-order transmission Jones matrix of eps(Ny,Nx) layer of thickness h.

    Light enters from a half-space eps_in, exits into eps_out.  Returns dict with
      J    : 2x2 power-normalised Jones matrix [[xx, xy],[yx, yy]] (rows=out, cols=in)
      T,R  : total transmitted / reflected power (all orders, incident zero-order)  for x and y input
    """
    Kx, Ky = _K(lam, px, py, Mx, My)
    N = Kx.size
    P, Q = _layer_P_Q(eps, Kx, Ky, Mx, My)
    Z = jnp.zeros_like(P)
    G = jnp.block([[Z, P], [Q, Z]])
    W0, W0i, _, _ = _medium(1.0, Kx, Ky)
    Wi_, Wii, kzi, Vi = _medium(eps_in, Kx, Ky)
    Wo, Woi, kzo, Vo = _medium(eps_out, Kx, Ky)
    S = _T2S(W0i @ expm(1j * (2 * jnp.pi * h / lam / 2**k) * G) @ W0)
    for _ in range(k):
        S = _star(S, S)
    S = _star(_star(_T2S(W0i @ Wi_), S), _T2S(Woi @ W0))
    R, _, T, _ = S
    n0 = My * (2 * Mx + 1) + Mx
    sel = jnp.array([n0, N + n0])
    J = jnp.sqrt(kzo[n0].real / kzi[n0].real) * T[jnp.ix_(sel, sel)]

    def flux(a, V):                                              # Re(Ex Hy* - Ey Hx*)
        E, H = a, V @ a
        return jnp.sum(jnp.real(E[:N] * jnp.conj(H[N:]) - E[N:] * jnp.conj(H[:N])))
    Tp = jnp.array([flux(T[:, i], Vo) / flux(jnp.eye(2 * N)[:, i], Vi) for i in sel])
    Rp = jnp.array([flux(R[:, i], Vi) / flux(jnp.eye(2 * N)[:, i], Vi) for i in sel])
    return {"J": J, "T": Tp, "R": Rp}


def _solve_cell(eps, h, lam, px, py, eps_in, eps_out, Mx, My, k):
    return solve(eps, h, lam, px, py, eps_in, eps_out, Mx=Mx, My=My, k=k)


def rect_eps(W, L, px, py, eps_p, eps_bg=1.0, Nx=128, Ny=128):
    """Centred W x L rectangle on an anti-aliased Nx x Ny pixel grid (differentiable in W, L)."""
    def cover(w, p, N):
        dx = p / N
        c = (jnp.arange(N) + 0.5 - N / 2) * dx
        lo, hi = jnp.maximum(c - dx / 2, -w / 2), jnp.minimum(c + dx / 2, w / 2)
        return jnp.clip(hi - lo, 0.0, dx) / dx
    f = cover(L, py, Ny)[:, None] * cover(W, px, Nx)[None, :]
    return eps_bg + (eps_p - eps_bg) * f
