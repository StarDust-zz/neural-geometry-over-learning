#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run neural-geometry-over-learning assertions. python3 experiment.py"""

from geometry import (
    XOR_CONDITIONS,
    cross_set_xor,
    decode_feature,
    early_code,
    hebbian_error,
    late_code,
    minimal_xor,
    optimal_spectrum,
    participation_ratio,
    random_mixed,
    shattering_score,
)


def _ok(cond, msg):
    if not cond:
        raise AssertionError(msg)


def scenario_1():
    """Optimal spectrum flattens as p grows (Wakhloo eq.)."""
    omega = (4.0, 2.0, 1.0)
    early = optimal_spectrum(omega, p=1.0)
    late = optimal_spectrum(omega, p=50.0)
    pr_e = participation_ratio(early)
    pr_l = participation_ratio(late)
    _ok(pr_e < pr_l, (pr_e, pr_l))
    # late ratios closer to 1
    r_e = early[0] / early[-1]
    r_l = late[0] / late[-1]
    _ok(r_e > r_l, (r_e, r_l))
    return {"PR_early": round(pr_e, 3), "PR_late": round(pr_l, 3), "spread_early": round(r_e, 3), "spread_late": round(r_l, 3)}


def scenario_2():
    """Early code: higher c, lower PR. Late: higher SSF, noise off-axis."""
    e = early_code().stats()
    l = late_code().stats()
    _ok(e["c"] > l["c"], (e["c"], l["c"]))
    _ok(e["PR"] < l["PR"], (e["PR"], l["PR"]))
    _ok(e["f"] < l["f"], (e["f"], l["f"]))
    _ok(l["s"] == float("inf") or l["s"] >= e["s"], (e["s"], l["s"]))
    return {"early": {k: (v if v != float("inf") else "inf") for k, v in e.items() if k != "s"} | {"s": "inf" if e["s"] == float("inf") else round(e["s"], 3)},
            "late": {k: (v if v != float("inf") else "inf") for k, v in l.items() if k != "s"} | {"s": "inf" if l["s"] == float("inf") else round(l["s"], 3)}}


def scenario_3():
    """Few-shot Hebbian: compressed early code can beat the expanded late code."""
    # Tasks: shatter on z1 (the informative latent). Few samples, noisy-ish.
    zs_train = [(1.0, 0.2), (1.0, -0.2), (-1.0, 0.2), (-1.0, -0.2)]
    ys = [1, 1, -1, -1]
    zs_test = [(0.8, 0.5), (0.8, -0.5), (-0.8, 0.5), (-0.8, -0.5)]
    early = early_code()
    late = late_code()
    e_err = hebbian_error([early.embed(z) for z in zs_train], ys, [early.embed(z) for z in zs_test], ys)
    l_err = hebbian_error([late.embed(z) for z in zs_train], ys, [late.embed(z) for z in zs_test], ys)
    # With this shatter, both should be perfect or early <= late (few-shot prefers c)
    _ok(e_err <= l_err, (e_err, l_err))
    return {"early_err": e_err, "late_err": l_err}


def scenario_4():
    """Many-shot / new latent: late factorized code reads z2; early compressed misses it."""
    # Shatter on the weak latent z2. Early code barely represents it.
    zs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    ys = [1, -1, 1, -1]
    extra = [(0.3, 1.0), (0.3, -1.0), (-0.3, 1.0), (-0.3, -1.0)]
    early = early_code()
    late = late_code()
    e_err = hebbian_error([early.embed(z) for z in zs], ys, [early.embed(z) for z in extra], ys)
    l_err = hebbian_error([late.embed(z) for z in zs], ys, [late.embed(z) for z in extra], ys)
    _ok(l_err < e_err, (e_err, l_err))
    _ok(l_err == 0.0, l_err)
    return {"early_err_z2": e_err, "late_err_z2": l_err}


def scenario_5():
    """Random mixed: color, shape, XOR all decode. Minimal: only XOR."""
    rnd = random_mixed()
    mn = minimal_xor()
    r_c, r_s, r_x = decode_feature(rnd, 0), decode_feature(rnd, 1), decode_feature(rnd, 2)
    m_c, m_s, m_x = decode_feature(mn, 0), decode_feature(mn, 1), decode_feature(mn, 2)
    _ok(r_c == 1.0 and r_s == 1.0 and r_x == 1.0, (r_c, r_s, r_x))
    _ok(m_x == 1.0, m_x)
    _ok(m_c == 0.5 and m_s == 0.5, (m_c, m_s))
    return {"random": {"color": r_c, "shape": r_s, "xor": r_x}, "minimal": {"color": m_c, "shape": m_s, "xor": m_x}}


def scenario_6():
    """Shattering dimensionality is higher for mixed than for minimal (Wójcik)."""
    rnd = shattering_score(random_mixed())
    mn = shattering_score(minimal_xor())
    _ok(rnd > mn, (rnd, mn))
    _ok(rnd == 1.0, rnd)
    _ok(mn < 1.0, mn)
    return {"shatter_random": rnd, "shatter_minimal": mn}


def scenario_7():
    """Second set: late (aligned) XOR readout transfers; early independent mix does not."""
    late_a = minimal_xor()
    late_b = minimal_xor()  # same construction, same axis family
    # early B is a reshuffled mix, not aligned
    early_b = random_mixed()
    aligned = cross_set_xor(late_a, late_b)
    unaligned = cross_set_xor(late_a, early_b)
    _ok(aligned == 1.0, aligned)
    _ok(unaligned < aligned, (unaligned, aligned))
    _ok(len(XOR_CONDITIONS) == 4, XOR_CONDITIONS)
    return {"aligned": aligned, "unaligned": unaligned}


def main():
    scenarios = [
        ("1 optimal spectrum flattens with p", scenario_1),
        ("2 early: high c, low PR; late: high SSF", scenario_2),
        ("3 few-shot: compressed code is enough", scenario_3),
        ("4 many-shot weak latent: need the expanded code", scenario_4),
        ("5 mixed decodes all; minimal decodes XOR only", scenario_5),
        ("6 shattering dim drops from mixed to minimal", scenario_6),
        ("7 second set aligns only after the late geometry", scenario_7),
    ]
    failed = 0
    for name, fn in scenarios:
        try:
            info = fn()
            print(f"PASS  {name}")
            for k, v in info.items():
                print(f"      {k}: {v}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(scenarios) - failed}/{len(scenarios)} passed")
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
