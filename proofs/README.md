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

## ConformalCoverage.lean (claim C2, fully proven)

A Lean 4 + Mathlib proof of the second load-bearing claim: the honesty layer's split-conformal
predictor (`src/honesty.py::conformal_regression_qhat`) has finite-sample, distribution-free
marginal coverage `>= 1 - alpha`, and the abstention gate is sound. **No `sorry`.**

The measure-theoretic phrase "exchangeability ⇒ uniform rank of the test score" is made concrete
as the *uniform test-index model*: given `n+1` distinct scores, each index is equally likely to
be the held-out test point, and coverage is the fraction of indices landing inside the conformal
set. The central fact is that this fraction is `min(k, n+1)/(n+1)` **independent of the actual
score values**, which is exactly why the guarantee is distribution-free.

- `level_ge` **(proven)**: the arithmetic floor `ceil((n+1)(1-alpha)) / (n+1) >= 1 - alpha`.
- `levelCount_le` **(proven)**: for `alpha >= 0`, the level count `k = ceil((n+1)(1-alpha))` never
  exceeds `n+1`.
- `belowCount_eq` **(proven)**: for strictly monotone (distinct) scores, the number of other
  scores below index `t` is exactly its rank `t`. The order-statistic content, via `Fin.card_Iio`.
- `coverage_count` **(proven)**: the covered-index count over the test-index model is
  `min(k, n+1)`, value-independent. This is the exchangeability fact that was the former `sorry`.
- `covProb_eq_fraction` **(proven)**: `covProb` genuinely equals that covered fraction for any
  distinct score family (ties the definition to the count).
- `uniformRankFloor` **(proven)**: `covProb >= k/(n+1)`. Chained with `level_ge` gives
  `marginal_coverage`: `covProb >= 1 - alpha` (Vovk 2005; Angelopoulos and Bates 2023, Thm 1).
- `report_is_powered` / `underpowered_abstains` **(proven)**: abstention soundness. Whenever the
  gate reports, the claimed effect met the minimum detectable effect at this `N`; below it, it
  abstains. "Abstention is a safety property, not a storytelling device" as a theorem.

**Status: fully verified.** `lake build` exits 0 with no `sorry`. `#print axioms` on
`marginal_coverage`, `coverage_count`, and `uniformRankFloor` reports only the three standard
Mathlib axioms `[propext, Classical.choice, Quot.sound]` (no `sorryAx`).

**Scope / honest caveat.** The model formalizes exchangeability as its finite symmetric
representative (distinct, sorted scores under the uniform test-index law), which is the standard
combinatorial route and captures the distribution-free content. It does not build a general
`MeasureTheory` probability space or handle the measure-zero tie set for continuous scores; those
are routine extensions, not gaps in the argument proven here.

### Verify

Toolchain: Lean 4.34.0 + Mathlib (not run in CI; it pulls Mathlib). From `proofs/`:

```bash
lake exe cache get
lake build            # exits 0; every theorem type-checks with no sorry
```

Verified against Mathlib on Lean 4.34.0 (`lake build` completed successfully, 3094 jobs, no
`sorry`; axioms limited to propext / Classical.choice / Quot.sound).
