#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Play the geometry path. python3 play.py"""

from geometry import (
    blend_alignment,
    blend_code,
    blend_selectivity,
    cross_set_feature,
    cross_set_xor,
    decode_feature,
    few_shot_distractor_error,
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
        "train sign(z1) on 2 points where z2 is 2x the label; test with z2 flipped.",
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


def main() -> None:
    print("neural geometry playground")
    print("a path, not a snapshot. stdlib only.")
    trap_path()
    alignment_path()
    abstraction_path()
    print()


if __name__ == "__main__":
    main()
