# Neural geometry over learning

A small, stdlib-only experiment: a population code is a **trajectory**, not a snapshot.

Two 2026 *Nature Neuroscience* papers name the same object from different ends.

- [Wakhloo, Slatton, Chung](https://doi.org/10.1038/s41593-025-02183-y) (4 Feb 2026, *Nat Neurosci* 29, 682–692, CC-BY 4.0). Four mesoscopic statistics govern linear-readout generalization on tasks that share a latent structure. Early optimal codes are lower dimensional and more correlated with the latents. Late codes expand and factorize. The optimal spectrum flattens with the number of samples `p`.
- [Wójcik et al.](https://doi.org/10.1038/s41593-026-02333-w) (*Nat Neurosci* 2026). Macaque PFC learning a new XOR rule starts high-dimensional, nonlinear, and randomly mixed, then becomes low-dimensional and rule-selective. A second stimulus set with the same structure realigns onto a shared axis so the old readout generalizes.

This pack is the discrete algebra of those claims. Not a replay of the recordings, not a trained net, not their +error bars.

```
python3 experiment.py
```

Last run: **7/7 passed**. Python 3.10+, stdlib only. No GPU, no network, no model API.

## Why this exists

People argue about whether PFC (or a hidden layer) “is” high-dimensional or low-dimensional. Both papers say that question is ill-posed. The geometry **moves with learning**. Chung et al. give the four numbers that a Hebbian readout actually sees. Wójcik et al. show the biological trajectory: mixed and shatterable, then minimal and abstract, then aligned across a new stimulus set.

The apparent fight over raw dimension is a snapshot artifact. Task-relevant geometry expands and factorizes. Irrelevant dichotomies (width, pure color, pure shape once the rule is known) collapse. You cannot read the stage off one cloud of points.

## The object

Wakhloo generalization error decreases in four stats (their eq. 1):

| symbol | name | early optimal | late optimal |
| --- | --- | --- | --- |
| `c` | neural–latent correlation | higher | lower |
| `f` | signal–signal factorization | lower | higher (orthogonal, whitened latents) |
| `s` | signal–noise factorization | — | noise off the coding directions |
| `PR(Ψ)` | participation ratio | lower | higher (spectrum flattens) |

Closed form for the optimal neural eigenvalues (their eq. 2):

```
psi_i = C * omega_i / (2 p omega_i + π Σ_k omega_k)
```

As `p` grows, `psi` flattens: the code starts paying rent on the weak latents.

Wójcik adds the stage names:

| stage | selectivity | linearly decodable | transfer |
| --- | --- | --- | --- |
| early | random mixed (color, shape, XOR) | all three | a new set does not share the axis |
| late | minimal (XOR only) | XOR | old readout fires on the new set |

## Run

```bash
python3 experiment.py
```

Seven `PASS` lines and `7/7 passed`.

| file | what it is |
| --- | --- |
| [`geometry.py`](geometry.py) | four stats, optimal spectrum, Hebbian readout, mixed vs minimal XOR |
| [`experiment.py`](experiment.py) | the seven assertions |
| [`DESCRIPTION.md`](DESCRIPTION.md) | short lab note |

## Assertions

1. **Spectrum flattens with `p`.** Optimal `psi` at `p=1` is more peaked than at `p=50`. Participation ratio rises.
2. **Early vs late linear codes.** Compressed early code has higher `c`, lower `PR`, lower SSF than an orthogonal late code. (SNF is infinite here: noise is constructed off-axis, which is the paper’s optimal regime.)
3. **Few-shot.** A shatter on the strong latent is solved by the compressed code. You do not need the expanded geometry yet.
4. **Weak latent.** A shatter on the second latent is missed by the compressed code and solved by the factorized one.
5. **Mixed vs minimal.** Random mixed selectivity linearly decodes color, shape, and XOR. Minimal XOR decodes XOR only (color and shape at chance).
6. **Shattering dimensionality drops.** Mean decode over those three dichotomies is 1.0 mixed, 2/3 minimal.
7. **Second set.** A late (minimal) XOR readout transfers to another minimal population. It does not transfer to an unaligned mixed population.

## What this is not

- Not Chung’s MLP / DeepLabCut / V4–IT / rat CA1 analyses, and not a claim we reproduced their `R²`.
- Not Wójcik’s two macaques, electrode advances, or trial-termination behavior.
- Not a new decoder. The Hebbian rule is theirs (eq. 3).
- Not a product integration.

Their code: [awakhloo/population_geometry_optimal_coding](https://github.com/awakhloo/population_geometry_optimal_coding), [m-j-wojcik/pfc_learning](https://github.com/m-j-wojcik/pfc_learning).

## Citation

```
@article{wakhloo2026geometry,
  title   = {Neural population geometry and optimal coding of tasks with shared latent structure},
  author  = {Wakhloo, Albert J. and Slatton, Will and Chung, SueYeon},
  journal = {Nature Neuroscience},
  volume  = {29},
  pages   = {682--692},
  year    = {2026},
  doi     = {10.1038/s41593-025-02183-y}
}

@article{wojcik2026pfc,
  title   = {Learning shapes neural geometry in the primate prefrontal cortex},
  author  = {W{\'o}jcik, Micha{\l} J. and Stroud, Jake P. and Wasmuht, Dante
             and Kusunoki, Makoto and Kadohisa, Mikiko and Buckley, Mark J.
             and Costa, Rui Ponte and Myers, Nicholas E. and Hunt, Laurence T.
             and Duncan, John and Stokes, Mark G.},
  journal = {Nature Neuroscience},
  year    = {2026},
  doi     = {10.1038/s41593-026-02333-w}
}
```

This repository is an independent discrete experiment. It is not affiliated with those authors.

## License

[MIT](LICENSE). Copyright (c) 2026 Igor Pistolyaka.

The Wakhloo paper is CC-BY 4.0. The Wójcik paper keeps its publisher terms. Nothing here is a copy of their codebases.
