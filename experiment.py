#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run neural-geometry-over-learning assertions. python3 experiment.py"""

from geometry import (
    XOR_CONDITIONS,
    blend_alignment,
    blend_code,
    blend_selectivity,
    cross_code_error,
    cross_set_feature,
    cross_set_xor,
    decode_feature,
    early_code,
    few_shot_distractor_error,
    distractor_error,
    hebbian_error,
    late_code,
    late_code_shared_z1,
    leaky_early_code,
    minimal_xor,
    NOISE_SEEDS,
    optimal_spectrum,
    optimal_spectrum_path,
    participation_ratio,
    random_mixed,
    shattering_score,
    weak_latent_error,
    xor_aligned_feature_mixed,
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
    e_err = weak_latent_error(early_code())
    l_err = weak_latent_error(late_code())
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


def scenario_8():
    """Early→late is a path: c falls, factorization and PR rise, weak latent unlocks."""
    ts = (0.0, 0.5, 1.0)
    stats = [blend_code(t).stats() for t in ts]
    _ok(stats[0]["c"] > stats[1]["c"] > stats[2]["c"], [s["c"] for s in stats])
    _ok(stats[0]["f"] < stats[1]["f"] < stats[2]["f"], [s["f"] for s in stats])
    _ok(stats[0]["PR"] < stats[1]["PR"] < stats[2]["PR"], [s["PR"] for s in stats])
    err0 = weak_latent_error(blend_code(0.0))
    err1 = weak_latent_error(blend_code(1.0))
    _ok(err0 > err1, (err0, err1))
    _ok(err1 == 0.0, err1)
    return {
        "c": [round(s["c"], 3) for s in stats],
        "f": [round(s["f"], 3) for s in stats],
        "PR": [round(s["PR"], 3) for s in stats],
        "z2_err_early": err0,
        "z2_err_late": err1,
    }


def scenario_9():
    """On-axis noise makes SNF finite; off-axis late noise stays clean."""
    leaky = leaky_early_code().stats()
    late = late_code().stats()
    early = early_code().stats()
    _ok(leaky["s"] != float("inf"), leaky["s"])
    _ok(late["s"] > leaky["s"], (leaky["s"], late["s"]))
    _ok(early["s"] > leaky["s"], (leaky["s"], early["s"]))
    return {"leaky_s": round(leaky["s"], 3), "early_s": "inf", "late_s": "inf"}


def scenario_10():
    """Shared latent structure: a z1 readout transfers; a remapped z2 readout does not."""
    src = late_code()
    dest = late_code_shared_z1()
    z1_train = [(1.0, 0.2), (1.0, -0.2), (-1.0, 0.2), (-1.0, -0.2)]
    y1 = [1, 1, -1, -1]
    z1_test = [(0.8, 0.5), (0.8, -0.5), (-0.8, 0.5), (-0.8, -0.5)]
    z2_train = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    y2 = [1, -1, 1, -1]
    z2_test = [(0.3, 1.0), (0.3, -1.0), (-0.3, 1.0), (-0.3, -1.0)]
    z1_xfer = cross_code_error(src, dest, z1_train, y1, z1_test)
    z2_xfer = cross_code_error(src, dest, z2_train, y2, z2_test)
    z2_same = cross_code_error(src, src, z2_train, y2, z2_test)
    _ok(z1_xfer == 0.0, z1_xfer)
    _ok(z2_xfer > z1_xfer, (z1_xfer, z2_xfer))
    _ok(z2_same == 0.0, z2_same)
    return {"z1_transfer_err": z1_xfer, "z2_transfer_err": z2_xfer, "z2_same_code_err": z2_same}


def scenario_11():
    """Mixed→minimal path: XOR stays decodable; irrelevant dichotomies collapse."""
    t0, t_mid, t1 = blend_selectivity(0.0), blend_selectivity(0.75), blend_selectivity(1.0)
    x0, x_mid, x1 = decode_feature(t0, 2), decode_feature(t_mid, 2), decode_feature(t1, 2)
    c0, c1 = decode_feature(t0, 0), decode_feature(t1, 0)
    sh0, sh_mid, sh1 = shattering_score(t0), shattering_score(t_mid), shattering_score(t1)
    _ok(x0 == 1.0 and x_mid == 1.0 and x1 == 1.0, (x0, x_mid, x1))
    _ok(c0 == 1.0 and c1 == 0.5, (c0, c1))
    _ok(sh0 > sh_mid > sh1, (sh0, sh_mid, sh1))
    return {
        "xor": [x0, x_mid, x1],
        "color_early": c0,
        "color_late": c1,
        "shatter": [round(sh0, 3), round(sh_mid, 3), round(sh1, 3)],
    }


def scenario_12():
    """Transfer is axis-specific: shared XOR axis transfers; remixed color does not."""
    late = minimal_xor()
    aligned = xor_aligned_feature_mixed()
    mixed = random_mixed()
    xor_aligned = cross_set_xor(late, aligned)
    xor_unaligned = cross_set_xor(late, mixed)
    color_local = decode_feature(aligned, 0)
    color_xfer = cross_set_feature(mixed, aligned, 0)
    _ok(xor_aligned == 1.0, xor_aligned)
    _ok(xor_unaligned < xor_aligned, (xor_unaligned, xor_aligned))
    _ok(color_local == 1.0, color_local)
    _ok(color_xfer < color_local, (color_xfer, color_local))
    return {
        "xor_aligned": xor_aligned,
        "xor_unaligned": xor_unaligned,
        "color_local_on_remix": color_local,
        "color_transfer_mixed_to_remix": color_xfer,
    }


def scenario_13():
    """p-sweep: optimal spectrum flattens monotonically as sample count grows."""
    omega = (4.0, 2.0, 1.0)
    ps = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    rows = optimal_spectrum_path(omega, ps)
    prs = [r["PR"] for r in rows]
    spreads = [r["spread"] for r in rows]
    weak = [r["weak_share"] for r in rows]
    _ok(all(prs[i] < prs[i + 1] for i in range(len(prs) - 1)), prs)
    _ok(all(spreads[i] > spreads[i + 1] for i in range(len(spreads) - 1)), spreads)
    _ok(all(weak[i] < weak[i + 1] for i in range(len(weak) - 1)), weak)
    _ok(prs[-1] > 2.9, prs[-1])
    _ok(spreads[-1] < 1.2, spreads[-1])
    return {
        f"p={int(r['p']) if float(r['p']).is_integer() else r['p']}": {
            "PR": round(r["PR"], 3),
            "spread": round(r["spread"], 3),
            "weak_share": round(r["weak_share"], 3),
            "psi": [round(x, 4) for x in r["psi"]],
        }
        for r in rows
    }


def scenario_14():
    """Few-shot trap: late fits a correlated distractor; unlocking z2 is an earlier t."""
    e_err = few_shot_distractor_error(early_code())
    l_err = few_shot_distractor_error(late_code())
    _ok(e_err == 0.0, e_err)
    _ok(l_err == 1.0, l_err)
    ts = tuple(i / 10 for i in range(11))
    trap = [few_shot_distractor_error(blend_code(t)) for t in ts]
    z2 = [weak_latent_error(blend_code(t)) for t in ts]
    unlocked = [t for t, e in zip(ts, z2) if e == 0.0]
    safe = [t for t, e in zip(ts, trap) if e == 0.0]
    _ok(unlocked and safe, (unlocked, safe))
    _ok(min(unlocked) < max(safe), (min(unlocked), max(safe)))
    _ok(any(z == 0.0 and tr == 0.0 for z, tr in zip(z2, trap)), list(zip(ts, z2, trap)))
    return {
        "early_trap_err": e_err,
        "late_trap_err": l_err,
        "z2_unlocks_at": min(unlocked),
        "trap_still_zero_until": max(safe),
    }


def scenario_15():
    """Frozen late XOR readout: transfer is 0 anti-aligned, 1 once the axis matches."""
    src = minimal_xor()
    ts = (0.0, 0.25, 0.5, 0.75, 1.0)
    xfer = [cross_set_xor(src, blend_alignment(t)) for t in ts]
    color = [decode_feature(blend_alignment(t), 0) for t in ts]
    _ok(xfer[0] == 0.0, xfer[0])
    _ok(xfer[-1] == 1.0, xfer[-1])
    _ok(all(xfer[i] <= xfer[i + 1] for i in range(len(xfer) - 1)), xfer)
    _ok(all(c == 1.0 for c in color), color)
    return {"xor_transfer": xfer, "color_local": color}


def scenario_16():
    """Frozen mixed readouts on the way to minimal: XOR rides along; color dies."""
    src = blend_selectivity(0.0)
    ts = (0.0, 0.25, 0.5, 0.75, 1.0)
    xor_xfer = [cross_set_xor(src, blend_selectivity(t)) for t in ts]
    color_xfer = [cross_set_feature(src, blend_selectivity(t), 0) for t in ts]
    _ok(all(x == 1.0 for x in xor_xfer), xor_xfer)
    _ok(color_xfer[0] == 1.0, color_xfer[0])
    _ok(color_xfer[-1] == 0.5, color_xfer[-1])
    _ok(color_xfer[0] > color_xfer[-1], color_xfer)
    return {"xor_transfer": xor_xfer, "color_transfer": color_xfer}


def scenario_17():
    """More aligned copies do not save late. One flipped pair does."""
    e2 = distractor_error(early_code(), n_aligned=2)
    l2 = distractor_error(late_code(), n_aligned=2)
    e8 = distractor_error(early_code(), n_aligned=8)
    l8 = distractor_error(late_code(), n_aligned=8)
    e_bal = distractor_error(early_code(), n_aligned=2, n_flipped=2)
    l_bal = distractor_error(late_code(), n_aligned=2, n_flipped=2)
    _ok(e2 == 0.0 and e8 == 0.0, (e2, e8))
    _ok(l2 == 1.0 and l8 == 1.0, (l2, l8))
    _ok(e_bal == 0.0, e_bal)
    _ok(l_bal == 0.0, l_bal)
    return {
        "aligned_n2": {"early": e2, "late": l2},
        "aligned_n8": {"early": e8, "late": l8},
        "plus_one_flipped_pair": {"early": e_bal, "late": l_bal},
    }


def scenario_18():
    """Observation noise blurs the trap; it does not flip who wins until the cloud is balanced."""
    scale = 0.5
    e2 = distractor_error(early_code(), n_aligned=2, noise=scale, seeds=NOISE_SEEDS)
    l2 = distractor_error(late_code(), n_aligned=2, noise=scale, seeds=NOISE_SEEDS)
    e8 = distractor_error(early_code(), n_aligned=8, noise=scale, seeds=NOISE_SEEDS)
    l8 = distractor_error(late_code(), n_aligned=8, noise=scale, seeds=NOISE_SEEDS)
    e_bal = distractor_error(early_code(), n_aligned=2, n_flipped=4, noise=scale, seeds=NOISE_SEEDS)
    l_bal = distractor_error(late_code(), n_aligned=2, n_flipped=4, noise=scale, seeds=NOISE_SEEDS)
    path = [distractor_error(blend_code(t), n_aligned=2, noise=scale, seeds=NOISE_SEEDS) for t in (0.0, 0.5, 1.0)]
    _ok(e2 < 0.05, e2)
    _ok(l2 > 0.7, l2)
    _ok(abs(l8 - l2) < 0.05, (l2, l8))
    _ok(e8 < 0.05, e8)
    _ok(l_bal < 0.05 and e_bal < 0.05, (e_bal, l_bal))
    _ok(path[0] < path[1] < path[2], path)
    return {
        "noise": scale,
        "aligned_n2": {"early": round(e2, 3), "late": round(l2, 3)},
        "aligned_n8": {"early": round(e8, 3), "late": round(l8, 3)},
        "balanced_n6": {"early": round(e_bal, 3), "late": round(l_bal, 3)},
        "path_n2": [round(x, 3) for x in path],
    }


def main():
    scenarios = [
        ("1 optimal spectrum flattens with p", scenario_1),
        ("2 early: high c, low PR; late: high SSF", scenario_2),
        ("3 few-shot: compressed code is enough", scenario_3),
        ("4 many-shot weak latent: need the expanded code", scenario_4),
        ("5 mixed decodes all; minimal decodes XOR only", scenario_5),
        ("6 shattering dim drops from mixed to minimal", scenario_6),
        ("7 second set aligns only after the late geometry", scenario_7),
        ("8 early→late path: c falls, f and PR rise", scenario_8),
        ("9 on-axis noise: SNF becomes finite", scenario_9),
        ("10 shared z1 transfers; remapped z2 does not", scenario_10),
        ("11 mixed→minimal: XOR stays, extras collapse", scenario_11),
        ("12 transfer is the shared axis, not the whole mix", scenario_12),
        ("13 p-sweep: optimal spectrum flattens with sample count", scenario_13),
        ("14 few-shot trap: late overfits z2; z2 unlocks earlier", scenario_14),
        ("15 frozen XOR: transfer walks with axis alignment", scenario_15),
        ("16 frozen mixed: XOR rides to minimal; color dies", scenario_16),
        ("17 more points: aligned copies fail; a flipped pair saves late", scenario_17),
        ("18 noise: trap blurs, more aligned copies still fail", scenario_18),
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
