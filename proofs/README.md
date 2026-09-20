# Formal proof: the neuroforecasting aggregation kernel

## NeuroforecastProof.lean

A Lean 4 + Mathlib proof of the statistical kernel behind the individual-to-aggregate
bridge in the value-decision stack (see `VALUE_DECISION_STACK.md`, "The aggregation
bridge"). It formalizes why a fixed neural/value signal correlates more strongly with an
aggregate outcome than with any individual outcome.

Model: individual outcome `Y = S + ε` with shared signal variance `a > 0` and
idiosyncratic-noise variance `v ≥ 0`; the squared correlation with the signal is
`rhoSq a v = a / (a + v)`, and averaging over `N` people replaces `v` by `v / N`.

Theorems:
- `rhoSq_antitone`: less idiosyncratic noise gives a strictly higher squared correlation.
- `aggregation_improves`: for any group size `N > 1`, the signal correlates strictly more
  with the `N`-averaged outcome than with one individual. This is the neuroforecasting kernel.
- `more_data_is_better`: the aggregate correlation strictly increases with sample size.
- `aggregate_below_one`: it stays below 1 for every finite sample (aggregation denoises but
  does not manufacture signal).

Scope: this proves the aggregation identity that makes neuroforecasting possible and
quantifies it (Genevsky and Knutson 2017; Knutson and Genevsky 2018). It does not prove that
any particular region is the signal, that the noise is independent, or that a given dataset
behaves this way. Those are empirical.

This is the correlation-form sibling of the inverse-variance fusion inequality in
behavioral_decoding (`proofs/FusionMath.lean`): both are the same variance-reduction
principle, one combining people, the other combining modalities.

### Verify

Not run in CI (it pulls Mathlib). To check locally, in a Lean project with mathlib as a
dependency and the cache fetched:

```bash
lake exe cache get
lake build            # exits 0 iff every theorem type-checks, no sorry
```

Verified against Mathlib on Lean 4.34.0 (`lake build` exit 0, no `sorry`).
