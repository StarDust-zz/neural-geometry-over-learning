# Neural geometry over learning

Discrete pack of two 2026 *Nature Neuroscience* claims: a population
code is a trajectory, not a snapshot.

```
python3 experiment.py
python3 play.py
```

No GPU. No recordings. No trained net.

## Sources

- Wakhloo, Slatton, Chung. *Neural population geometry and optimal
  coding of tasks with shared latent structure.* Nat Neurosci 29,
  682–692 (published 4 Feb 2026).
  https://doi.org/10.1038/s41593-025-02183-y (CC-BY 4.0)
- Wójcik et al. *Learning shapes neural geometry in the primate
  prefrontal cortex.* Nat Neurosci (2026).
  https://doi.org/10.1038/s41593-026-02333-w

## The object

Four mesoscopic stats (Wakhloo) govern Hebbian readout error on
tasks that share latents: neural–latent correlation `c`,
signal–signal factorization `f`, signal–noise factorization `s`,
participation ratio `PR`. Early optimal codes are tighter and more
correlated. Late codes expand and factorize. The closed-form
spectrum flattens with sample count `p`.

Wójcik: the same PFC starts randomly mixed and high-dimensional
(color, shape, XOR all linearly decodable), then becomes minimal
and rule-selective (XOR only). A second stimulus set with the same
structure realigns onto a shared axis.

Apparent tension (raw dimension up vs down) is not a contradiction:
task-relevant geometry expands and factorizes; irrelevant dichotomies
collapse. A snapshot cannot tell you which stage you are in.

## Keepers

If 21/21 pass: **the code is a scheduled change in geometry.** Early
and late are different executables over the same latents, and the
stats move continuously along that path. The optimal spectrum
flattens with sample count `p`: PR rises, spread falls, the weak
mode gains share. Mixed vs minimal is a learning stage, not a
rival architecture. Transfer is alignment of a shared axis (a
latent, or XOR), not a new decoder and not the whole mix.
On-axis noise is the thing a late code gets rid of.

The few-shot trap is the practical bite: unlocking a weak latent
and becoming distractible are different times on the same path.
A frozen mixed XOR readout survives abstraction; a frozen color
readout does not. More aligned copies do not save the late code;
one flipped pair does. Observation noise blurs the cliff. It does
not replace a balanced sample. Sequential arrival: late stays
trapped until a flipped pair arrives, can be re-trapped by a
biased tail, and an EMA recovers faster than the batch then
overshoots into the opposite trap. Early is immune to order.
