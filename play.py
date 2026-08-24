#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Play the geometry path. python3 play.py"""

from geometry import (
    NOISE_SEEDS,
    blend_alignment,
    blend_code,
    blend_selectivity,
    cross_set_feature,
    cross_set_xor,
    decode_feature,
    distractor_error,
    few_shot_distractor_error,
    early_code,
    late_code,
    minimal_xor,
    shattering_score,
    weak_latent_error,
)


def _bar(x: float, width: int = 14) -> str:
    x = max(0.0, min(1.0, float(x)))
    n = int(round(x * width))
    return "#" * n + "." * (width - n)


def _section(title: str, note: str) -> None:
    print()
    print(title)
    print(note)


def trap_path() -> None:
    _section(
        "A  few-shot trap",
        "train sign(z1) on 2 aligned points; test 10 points with z2 flipped.",
    )
    print(f"  {'t':>4}  {'trap':>4}  {'z2':>4}  {'c':>5}  {'f':>5}  {'PR':>5}  trap                 z2")
    window = []
    for i in range(11):
        t = i / 10
        code = blend_code(t)
        trap = few_shot_distractor_error(code)
        z2 = weak_latent_error(code)
        s = code.stats()
        mark = ""
        if trap == 0.0 and z2 == 0.0:
            mark = "  both ok"
            window.append(t)
        elif trap == 1.0:
            mark = "  overfit distractor"
        elif z2 > 0:
            mark = "  misses z2"
        print(
            f"  {t:4.1f}  {trap:4.2f}  {z2:4.2f}  {s['c']:5.3f}  {s['f']:5.3f}  {s['PR']:5.3f}  "
            f"{_bar(trap)}  {_bar(1.0 - z2)}{mark}"
        )
    lo, hi = min(window), max(window)
    print(f"  notice: z2 unlocks before the trap. useful window t in [{lo:.1f}, {hi:.1f}].")


def alignment_path() -> None:
    _section(
        "B  frozen XOR readout, second set walks onto the axis",
        "w trained on minimal XOR. dest goes anti-aligned -> aligned remix.",
    )
    src = minimal_xor()
    print(f"  {'a':>4}  xfer  color  shatter")
    for i in range(9):
        t = i / 8
        dest = blend_alignment(t)
        xfer = cross_set_xor(src, dest)
        color = decode_feature(dest, 0)
        shatter = shattering_score(dest)
        print(
            f"  {t:4.2f}  {xfer:4.2f}  {color:5.2f}  {shatter:7.2f}  "
            f"{_bar(xfer)}  xfer"
        )
    print("  notice: color stays locally decodable. transfer is the shared sign, not richness.")


def abstraction_path() -> None:
    _section(
        "C  freeze the mixed readouts while the population goes minimal",
        "same w, moving geometry. XOR was already in the mix; color is discarded.",
    )
    src = blend_selectivity(0.0)
    print(f"  {'t':>4}  xor   color")
    for i in range(9):
        t = i / 8
        dest = blend_selectivity(t)
        xor = cross_set_xor(src, dest)
        color = cross_set_feature(src, dest, 0)
        print(
            f"  {t:4.2f}  {xor:4.2f}  {color:5.2f}  "
            f"{_bar(xor)}  {_bar(color)}"
        )
    print("  notice: abstraction is lossy. the rule rides along; discarded features do not.")


def more_points() -> None:
    _section(
        "D  more points",
        "same trap test (z2 flipped). extra aligned copies vs one flipped pair.",
    )
    print(f"  {'cloud':<22}  early  late")
    rows = (
        ("aligned n=2", 2, 0),
        ("aligned n=4", 4, 0),
        ("aligned n=8", 8, 0),
        ("n=2 + 1 flipped pair", 2, 2),
        ("n=2 + 2 flipped pairs", 2, 4),
        ("n=2 + 3 flipped pairs", 2, 6),
    )
    for name, na, nf in rows:
        e = distractor_error(early_code(), n_aligned=na, n_flipped=nf)
        l = distractor_error(late_code(), n_aligned=na, n_flipped=nf)
        print(f"  {name:<22}  {e:5.2f}  {l:5.2f}  {_bar(e)}  {_bar(l)}")
    print("  notice: more of the same spurious correlation does not save late. one flipped pair does.")


def noise_sweep() -> None:
    _section(
        "E  observation noise (mean over 8 seeds, 10 test points)",
        "additive ambient jitter. aligned n=2 / aligned n=8 / balanced n=6.",
    )
    print(f"  {'s':>4}  {'e2':>5}  {'l2':>5}  {'e8':>5}  {'l8':>5}  {'e_bal':>5}  {'l_bal':>5}")
    for scale in (0.0, 0.25, 0.5, 1.0, 1.5):
        seeds = (7,) if scale == 0.0 else NOISE_SEEDS
        e2 = distractor_error(early_code(), n_aligned=2, noise=scale, seeds=seeds)
        l2 = distractor_error(late_code(), n_aligned=2, noise=scale, seeds=seeds)
        e8 = distractor_error(early_code(), n_aligned=8, noise=scale, seeds=seeds)
        l8 = distractor_error(late_code(), n_aligned=8, noise=scale, seeds=seeds)
        eb = distractor_error(early_code(), n_aligned=2, n_flipped=4, noise=scale, seeds=seeds)
        lb = distractor_error(late_code(), n_aligned=2, n_flipped=4, noise=scale, seeds=seeds)
        print(
            f"  {scale:4.2f}  {e2:5.2f}  {l2:5.2f}  {e8:5.2f}  {l8:5.2f}  {eb:5.2f}  {lb:5.2f}"
        )
    print()
    print("  t-path at s=0.5, aligned n=2 (trap rises) vs balanced n=6 (stays down)")
    print(f"  {'t':>4}  n2     bal")
    for i in range(11):
        t = i / 10
        a = distractor_error(blend_code(t), n_aligned=2, noise=0.5, seeds=NOISE_SEEDS)
        b = distractor_error(blend_code(t), n_aligned=2, n_flipped=4, noise=0.5, seeds=NOISE_SEEDS)
        print(f"  {t:4.1f}  {a:5.2f}  {b:5.2f}  {_bar(a)}  {_bar(b)}")
    print("  notice: noise blurs the cliff. it does not replace a flipped pair.")


def main() -> None:
    print("neural geometry playground")
    print("a path, not a snapshot. stdlib only.")
    trap_path()
    alignment_path()
    abstraction_path()
    more_points()
    noise_sweep()
    print()


if __name__ == "__main__":
    main()
