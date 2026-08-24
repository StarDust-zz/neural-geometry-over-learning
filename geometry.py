# SPDX-License-Identifier: MIT
"""Neural geometry over learning — discrete pack.

Stdlib only. No GPU, no network, no product imports.

Claims under test:

  Wakhloo, Slatton, Chung, Nat Neurosci 29, 682–692 (4 Feb 2026).
  https://doi.org/10.1038/s41593-025-02183-y  (CC-BY 4.0)
  Four mesoscopic stats (c, f, s, PR) govern linear-readout generalization
  on tasks that share a latent structure. Early optimal codes are lower
  dimensional and more correlated with latents; late codes expand and
  factorize. Optimal spectrum flattens with sample count p:
      psi_i = C * omega_i / (2 p omega_i + pi * sum omega).

  Wójcik et al., Nat Neurosci (2026).
  https://doi.org/10.1038/s41593-026-02333-w
  Macaque PFC learning an XOR rule starts high-dimensional, nonlinear,
  randomly mixed, then becomes low-dimensional and rule-selective.
  A second stimulus set with the same structure realigns onto a shared
  axis so the old readout generalizes.

This pack is the discrete algebra of those claims, not a replay of
the recordings or a trained net.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _scale(v: Sequence[float], s: float) -> list[float]:
    return [s * x for x in v]


def participation_ratio(evals: Sequence[float]) -> float:
    """PR(Ψ) = (Tr Ψ)^2 / Tr(Ψ^2) for a diagonal (or already-eigenspanned) code."""
    pos = [max(x, 0.0) for x in evals]
    tr = sum(pos)
    tr2 = sum(x * x for x in pos)
    if tr2 <= 0:
        return 0.0
    return (tr * tr) / tr2


def optimal_spectrum(omega: Sequence[float], p: float, C: float = 1.0) -> list[float]:
    """Wakhloo et al. eq. for the optimal neural covariance eigenvalues."""
    s = sum(omega)
    return [C * w / (2.0 * p * w + math.pi * s) for w in omega]


def optimal_spectrum_path(
    omega: Sequence[float], ps: Sequence[float], C: float = 1.0
) -> list[dict[str, float | list[float]]]:
    """Optimal ψ, PR, peak-to-tail spread, and weak-mode share vs sample count p."""
    rows: list[dict[str, float | list[float]]] = []
    for p in ps:
        psi = optimal_spectrum(omega, p, C)
        tr = sum(psi)
        rows.append(
            {
                "p": p,
                "psi": psi,
                "PR": participation_ratio(psi),
                "spread": psi[0] / psi[-1] if psi[-1] else float("inf"),
                "weak_share": psi[-1] / tr if tr else 0.0,
            }
        )
    return rows


def neural_latent_corr(phi_cols: Sequence[Sequence[float]], psi_evals: Sequence[float], omega: Sequence[float]) -> float:
    """c = Tr(Φ Φ^T) / (Tr Ψ Tr Ω). phi_cols[k] is the k-th column of Φ (n,)."""
    tr_phiphi = sum(_dot(col, col) for col in phi_cols)
    tr_psi = sum(psi_evals)
    tr_omega = sum(omega)
    if tr_psi <= 0 or tr_omega <= 0:
        return 0.0
    return tr_phiphi / (tr_psi * tr_omega)


def signal_signal_factorization(phi_cols: Sequence[Sequence[float]], omega: Sequence[float]) -> float:
    """
    Discrete SSF: 1 when latent axes in neural space are orthogonal and
    equally scaled; smaller when they overlap. Matches the paper's 'independent
    latents on uncorrelated, whitened directions' without a full Ω^{-1} multiply.
    """
    if len(phi_cols) < 2:
        return 1.0
    norms = [math.sqrt(_dot(c, c)) for c in phi_cols]
    if any(n <= 0 for n in norms):
        return 0.0
    cosines = []
    for i in range(len(phi_cols)):
        for j in range(i + 1, len(phi_cols)):
            cosines.append(abs(_dot(phi_cols[i], phi_cols[j])) / (norms[i] * norms[j]))
    overlap = sum(cosines) / len(cosines)
    mean_n = sum(norms) / len(norms)
    balance = 1.0 - (sum(abs(n - mean_n) for n in norms) / (len(norms) * mean_n))
    return max(0.0, (1.0 - overlap) * balance)


def signal_noise_factorization(signal_dirs: Sequence[Sequence[float]], noise_dirs: Sequence[Sequence[float]]) -> float:
    """s high when noise is orthogonal to signal directions."""
    if not noise_dirs:
        return float("inf")
    acc = []
    for n in noise_dirs:
        nn = math.sqrt(_dot(n, n)) or 1.0
        leaks = []
        for s in signal_dirs:
            sn = math.sqrt(_dot(s, s)) or 1.0
            leaks.append(abs(_dot(n, s)) / (nn * sn))
        acc.append(max(leaks) if leaks else 0.0)
    leak = sum(acc) / len(acc)
    if leak <= 1e-12:
        return float("inf")
    return 1.0 / leak


def hebbian_error(xs: Sequence[Sequence[float]], ys: Sequence[int], x_test: Sequence[Sequence[float]], y_test: Sequence[int]) -> float:
    """Supervised Hebbian readout (Wakhloo eq. 3): w = mean_μ y_μ x_μ."""
    dim = len(xs[0])
    w = [0.0] * dim
    p = len(xs)
    for x, y in zip(xs, ys):
        for i, xi in enumerate(x):
            w[i] += y * xi
    w = _scale(w, 1.0 / p)
    wrong = 0
    for x, y in zip(x_test, y_test):
        pred = 1 if _dot(w, x) >= 0 else -1
        if pred != y:
            wrong += 1
    return wrong / len(y_test)


@dataclass(frozen=True)
class Code:
    """A linear population code x = A z, plus optional noise directions."""

    axes: tuple[tuple[float, ...], ...]  # d axes in R^n, columns of A
    omega: tuple[float, ...]
    noise: tuple[tuple[float, ...], ...] = ()

    @property
    def n(self) -> int:
        return len(self.axes[0])

    def embed(self, z: Sequence[float]) -> list[float]:
        x = [0.0] * self.n
        for axis, zi in zip(self.axes, z):
            for i, a in enumerate(axis):
                x[i] += a * zi
        return x

    def _psi(self) -> list[list[float]]:
        n = self.n
        psi = [[0.0] * n for _ in range(n)]
        for axis, w in zip(self.axes, self.omega):
            for i in range(n):
                for j in range(n):
                    psi[i][j] += axis[i] * w * axis[j]
        for d in self.noise:
            for i in range(n):
                for j in range(n):
                    psi[i][j] += d[i] * d[j]
        return psi

    def stats(self) -> dict[str, float]:
        phi = [tuple(_scale(axis, w)) for axis, w in zip(self.axes, self.omega)]
        psi = self._psi()
        tr_psi = sum(psi[i][i] for i in range(self.n))
        tr_psi2 = sum(psi[i][j] * psi[j][i] for i in range(self.n) for j in range(self.n))
        pr = (tr_psi * tr_psi) / tr_psi2 if tr_psi2 else 0.0
        return {
            "c": neural_latent_corr(phi, [tr_psi], self.omega),
            "f": signal_signal_factorization(self.axes, self.omega),
            "s": signal_noise_factorization(self.axes, self.noise),
            "PR": pr,
        }


def early_code() -> Code:
    """Compress: almost all variance on the informative latent."""
    return Code(
        axes=((1.0, 0.15, 0.0), (0.05, 0.02, 0.0)),
        omega=(4.0, 1.0),
        noise=((0.0, 0.0, 1.0),),
    )


def late_code() -> Code:
    """Expand and factorize: orthogonal, closer-to-equal axes. Noise off-signal."""
    return Code(
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        omega=(4.0, 1.0),
        noise=((0.0, 0.0, 1.0),),
    )


def _blend_axes(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], t: float
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple((1.0 - t) * x + t * y for x, y in zip(u, v)) for u, v in zip(a, b))


def blend_code(t: float) -> Code:
    """Linear path from early (t=0) to late (t=1). Same latents, moving geometry."""
    e, late = early_code(), late_code()
    return Code(axes=_blend_axes(e.axes, late.axes, t), omega=e.omega, noise=e.noise)


def leaky_early_code() -> Code:
    """Early compression with noise leaking onto the coding plane (finite SNF)."""
    e = early_code()
    return Code(axes=e.axes, omega=e.omega, noise=((0.4, 0.7, 0.2),))


# z2 = 2 * sign(z1) (aligned) or the flip. Label is always sign(z1).
_ALIGNED_Z = (
    (1.0, 2.0),
    (-1.0, -2.0),
    (0.8, 1.6),
    (-0.8, -1.6),
    (1.2, 2.4),
    (-1.2, -2.4),
    (0.6, 1.2),
    (-0.6, -1.2),
)
_FLIPPED_Z = (
    (1.0, -2.0),
    (-1.0, 2.0),
    (0.8, -1.6),
    (-0.8, 1.6),
    (1.2, -2.4),
    (-1.2, 2.4),
    (0.6, -1.2),
    (-0.6, 1.2),
)
_DISTRACTOR_TEST_Z = (
    (1.0, -1.0),
    (0.8, -0.8),
    (1.2, -1.2),
    (0.6, -0.6),
    (0.9, -1.5),
    (-1.0, 1.0),
    (-0.8, 0.8),
    (-1.2, 1.2),
    (-0.6, 0.6),
    (-0.9, 1.5),
)
NOISE_SEEDS = (3, 7, 11, 19, 29, 41, 43, 53)


def _label_z1(z: Sequence[float]) -> int:
    return 1 if z[0] > 0 else -1


def _lcg_signed(seed: int):
    s = seed & 0x7FFFFFFF

    def nxt() -> float:
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return (s / 0x7FFFFFFF) * 2.0 - 1.0

    return nxt


def jitter(vec: Sequence[float], scale: float, seed: int) -> list[float]:
    """Deterministic additive noise in ambient coordinates. scale=0 is a no-op."""
    if scale <= 0:
        return list(vec)
    nxt = _lcg_signed(seed)
    return [x + scale * nxt() for x in vec]


def distractor_cloud(n_aligned: int, n_flipped: int = 0) -> tuple[list[tuple[float, float]], list[int]]:
    """Training cloud for sign(z1). Aligned points have z2 tracking the label."""
    if n_aligned > len(_ALIGNED_Z) or n_flipped > len(_FLIPPED_Z):
        raise ValueError("distractor catalog exhausted")
    zs = list(_ALIGNED_Z[:n_aligned]) + list(_FLIPPED_Z[:n_flipped])
    return zs, [_label_z1(z) for z in zs]


def distractor_error(
    code: Code,
    n_aligned: int = 2,
    n_flipped: int = 0,
    noise: float = 0.0,
    seeds: Sequence[int] = (7,),
) -> float:
    """Mean Hebbian error on z2-flipped tests. noise>0 averages over `seeds`."""
    zs, ys = distractor_cloud(n_aligned, n_flipped)
    y_test = [_label_z1(z) for z in _DISTRACTOR_TEST_Z]
    acc = []
    for s0 in seeds:
        xs = [jitter(code.embed(z), noise, (s0 + 17 * i) & 0x7FFFFFFF) for i, z in enumerate(zs)]
        xt = [
            jitter(code.embed(z), noise, (10_000 + s0 + 17 * i) & 0x7FFFFFFF)
            for i, z in enumerate(_DISTRACTOR_TEST_Z)
        ]
        acc.append(hebbian_error(xs, ys, xt, y_test))
    return sum(acc) / len(acc)


def few_shot_distractor_error(code: Code) -> float:
    """Two aligned points, no noise: the original trap."""
    return distractor_error(code, n_aligned=2, n_flipped=0, noise=0.0, seeds=(7,))


def weak_latent_error(code: Code) -> float:
    """Shatter on the weak latent z2 (same points as the many-shot assertion)."""
    zs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    ys = (1, -1, 1, -1)
    extra = ((0.3, 1.0), (0.3, -1.0), (-0.3, 1.0), (-0.3, -1.0))
    return hebbian_error([code.embed(z) for z in zs], ys, [code.embed(z) for z in extra], ys)


def late_code_shared_z1() -> Code:
    """Same z1 axis as late_code; z2 is rotated onto the former noise direction."""
    return Code(
        axes=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        omega=(4.0, 1.0),
        noise=((0.0, 1.0, 0.0),),
    )


def cross_code_error(
    train: Code,
    test: Code,
    zs_train: Sequence[Sequence[float]],
    ys: Sequence[int],
    zs_test: Sequence[Sequence[float]],
) -> float:
    """Hebbian readout trained on one code, tested on another (same latents)."""
    return hebbian_error([train.embed(z) for z in zs_train], ys, [test.embed(z) for z in zs_test], ys)


# --- Wójcik XOR population ---

XOR_CONDITIONS = (
    # (color, shape, xor) with xor = color XOR shape, color/shape in {0,1}
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
)


def rates(selectivity: Sequence[tuple[float, float, float]], cond: tuple[int, int, int]) -> list[float]:
    c, s, x = cond
    out = []
    for bc, bs, bx in selectivity:
        out.append(bc * (2 * c - 1) + bs * (2 * s - 1) + bx * (2 * x - 1))
    return out


def decode_feature(selectivity: Sequence[tuple[float, float, float]], which: int) -> float:
    """Linear population decode of color=0 / shape=1 / xor=2. Accuracy in [0, 1]."""
    # Hebbian on the four conditions
    xs = [rates(selectivity, cond) for cond in XOR_CONDITIONS]
    ys = [1 if cond[which] == 1 else -1 for cond in XOR_CONDITIONS]
    w = [0.0] * len(selectivity)
    for x, y in zip(xs, ys):
        for i, xi in enumerate(x):
            w[i] += y * xi
    correct = 0
    for x, y in zip(xs, ys):
        pred = 1 if _dot(w, x) >= 0 else -1
        if pred == y:
            correct += 1
    return correct / 4.0


def random_mixed(n: int = 12) -> list[tuple[float, float, float]]:
    """Spherical mixed selectivity (deterministic LCG, no import random)."""
    seed = 1
    out = []
    for _ in range(n):
        trip = []
        for _k in range(3):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            trip.append((seed / 0x7FFFFFFF) * 2.0 - 1.0)
        out.append((trip[0], trip[1], trip[2]))
    return out


def minimal_xor(n: int = 12) -> list[tuple[float, float, float]]:
    seed = 99
    out = []
    for _ in range(n):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        sign = 1.0 if seed % 2 == 0 else -1.0
        out.append((0.0, 0.0, sign))
    return out


def blend_selectivity(t: float, n: int = 12) -> list[tuple[float, float, float]]:
    """Path from random mixed (t=0) to minimal XOR (t=1)."""
    mixed, minimal = random_mixed(n), minimal_xor(n)
    return [tuple((1.0 - t) * a + t * b for a, b in zip(r, m)) for r, m in zip(mixed, minimal)]


def xor_aligned_feature_mixed(n: int = 12) -> list[tuple[float, float, float]]:
    """Second set: same XOR signs as minimal_xor, remixed color/shape."""
    seed = 1
    out = []
    for _bc, _bs, bx in minimal_xor(n):
        trip = []
        for _k in range(2):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            trip.append((seed / 0x7FFFFFFF) * 2.0 - 1.0)
        out.append((trip[0], trip[1], bx))
    return out


def xor_anti_aligned_feature_mixed(n: int = 12) -> list[tuple[float, float, float]]:
    """Same remixed color/shape as the aligned second set; XOR signs flipped."""
    return [(bc, bs, -bx) for bc, bs, bx in xor_aligned_feature_mixed(n)]


def blend_alignment(t: float, n: int = 12) -> list[tuple[float, float, float]]:
    """Second set walks from anti-aligned XOR (t=0) onto the late XOR axis (t=1)."""
    anti, aligned = xor_anti_aligned_feature_mixed(n), xor_aligned_feature_mixed(n)
    return [tuple((1.0 - t) * a + t * b for a, b in zip(u, v)) for u, v in zip(anti, aligned)]


def shattering_score(selectivity: Sequence[tuple[float, float, float]]) -> float:
    """Mean linear-decode accuracy over color, shape, XOR (three dichotomies)."""
    return sum(decode_feature(selectivity, k) for k in (0, 1, 2)) / 3.0


def cross_set_feature(
    sel_a: Sequence[tuple[float, float, float]],
    sel_b: Sequence[tuple[float, float, float]],
    which: int,
) -> float:
    """Train a linear readout of feature `which` on A, test on B."""
    xs_a = [rates(sel_a, cond) for cond in XOR_CONDITIONS]
    ys = [1 if cond[which] == 1 else -1 for cond in XOR_CONDITIONS]
    w = [0.0] * len(sel_a)
    for x, y in zip(xs_a, ys):
        for i, xi in enumerate(x):
            w[i] += y * xi
    xs_b = [rates(sel_b, cond) for cond in XOR_CONDITIONS]
    correct = 0
    for x, y in zip(xs_b, ys):
        pred = 1 if _dot(w, x) >= 0 else -1
        if pred == y:
            correct += 1
    return correct / 4.0


def cross_set_xor(
    sel_a: Sequence[tuple[float, float, float]],
    sel_b: Sequence[tuple[float, float, float]],
) -> float:
    """Train XOR readout on population A, test on B (same condition order)."""
    return cross_set_feature(sel_a, sel_b, 2)
