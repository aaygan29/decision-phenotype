import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic

/-!
# Split-conformal coverage and the abstention gate, formally

This file proves the second load-bearing mathematical claim under the decision
phenotype (claim **C2** in `README.md`): the honesty layer's split-conformal
predictor has finite-sample, distribution-free marginal coverage `≥ 1 - α`, and
the abstention gate is *sound*.

It mirrors `NeuroforecastProof.lean`. The measure-theoretic phrase
"exchangeability ⇒ uniform rank of the test score" is made concrete here as the
**uniform test-index model**: given `n+1` distinct scores, each index is equally
likely to be the held-out test point, and coverage is the fraction of indices
that land inside the conformal set. The central fact (`coverage_count`) is that
this fraction is `min(k, n+1)/(n+1)` *independent of the actual score values* —
which is exactly why the guarantee is distribution-free. No `sorry`.

## The algorithm being verified (`src/honesty.py::conformal_regression_qhat`)

Given `n` exchangeable calibration nonconformity scores `s₁,…,sₙ = |residual|`
and a test score `s_{n+1}`, set the level `k = ⌈(n+1)(1-α)⌉` and take `q̂` to be
the `k`-th smallest calibration score. The prediction set is `{y : score(y) ≤ q̂}`.
For distinct reals, the coverage event `s_{n+1} ≤ q̂` is equivalent to "strictly
fewer than `k` calibration scores lie below `s_{n+1}`" — the counting form used
below as `belowCount t < k`. The guarantee (Vovk 2005; Angelopoulos & Bates
2023, Thm 1) is `P(s_{n+1} ≤ q̂) ≥ ⌈(n+1)(1-α)⌉ / (n+1) ≥ 1 - α`.
-/

namespace Conformal

open Finset

variable {α : ℝ}

/-- The conformal level count `k = ⌈(n+1)(1-α)⌉`, the index used in
`conformal_regression_qhat` (`ceil((n+1)*(1-alpha))`). -/
noncomputable def levelCount (n : ℕ) (α : ℝ) : ℕ :=
  ⌈((n : ℝ) + 1) * (1 - α)⌉₊

/-- **Deterministic core.** The coverage floor `k/(n+1)` is at least `1 - α`.
(The `α ≥ 0` hypothesis is kept for a uniform API with the theorems below; the
ceiling bound itself holds for any `α`.) -/
theorem level_ge (n : ℕ) (_hα : 0 ≤ α) :
    (1 - α) ≤ (levelCount n α : ℝ) / ((n : ℝ) + 1) := by
  have hpos : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  rw [le_div_iff₀ hpos]
  calc
    (1 - α) * ((n : ℝ) + 1) = ((n : ℝ) + 1) * (1 - α) := by ring
    _ ≤ (levelCount n α : ℝ) := by
          unfold levelCount; exact Nat.le_ceil _

/-- For `α ≥ 0`, the level count never exceeds the total sample `n+1`. -/
theorem levelCount_le (n : ℕ) (hα : 0 ≤ α) : levelCount n α ≤ n + 1 := by
  unfold levelCount
  have h1 : (1 : ℝ) - α ≤ 1 := by linarith
  have h2 : (0 : ℝ) ≤ (n : ℝ) + 1 := by positivity
  have hle : ((n : ℝ) + 1) * (1 - α) ≤ ((n + 1 : ℕ) : ℝ) := by
    have := mul_le_of_le_one_right h2 h1
    push_cast; linarith
  exact Nat.ceil_le.mpr hle

/-!
## The uniform test-index model

`belowCount v t` counts how many of the *other* `n` scores lie strictly below the
score at the held-out index `t`. The coverage event at `t` is `belowCount v t < k`
(equivalently `v t ≤ q̂` for distinct scores). We take the scores strictly
monotone (`StrictMono v`); exchangeability means coverage depends only on the
multiset of scores, so the sorted representative loses no generality.
-/

/-- Number of scores, other than `t`, that fall strictly below the score at `t`. -/
noncomputable def belowCount {n : ℕ} (v : Fin (n + 1) → ℝ) (t : Fin (n + 1)) : ℕ :=
  (univ.filter (fun i => i ≠ t ∧ v i < v t)).card

/-- For distinct (strictly monotone) scores, `belowCount` at index `t` is exactly
the rank of `t`, i.e. `t.val`. This is the order-statistic content. -/
theorem belowCount_eq {n : ℕ} {v : Fin (n + 1) → ℝ} (hv : StrictMono v)
    (t : Fin (n + 1)) : belowCount v t = (t : ℕ) := by
  unfold belowCount
  have hset : (univ.filter (fun i => i ≠ t ∧ v i < v t))
      = (univ.filter (fun i => i < t)) := by
    apply filter_congr
    intro i _
    constructor
    · rintro ⟨_, hlt⟩; exact hv.lt_iff_lt.mp hlt
    · intro hlt; exact ⟨ne_of_lt hlt, hv hlt⟩
  rw [hset]
  have hIio : (univ.filter (fun i : Fin (n + 1) => i < t)) = Iio t := by
    ext i; simp [mem_Iio]
  rw [hIio, Fin.card_Iio]

/-- Count of indices below a nat bound inside `Fin (n+1)` is `min k (n+1)`. -/
theorem card_lt_bound (n k : ℕ) :
    (univ.filter (fun t : Fin (n + 1) => (t : ℕ) < k)).card = min k (n + 1) := by
  classical
  rw [← Finset.card_map ⟨Fin.val, Fin.val_injective⟩]
  have hmap : (univ.filter (fun t : Fin (n + 1) => (t : ℕ) < k)).map
        ⟨Fin.val, Fin.val_injective⟩
      = (range (n + 1)).filter (· < k) := by
    ext m
    simp only [mem_map, mem_filter, mem_univ, true_and, Function.Embedding.coeFn_mk,
      mem_range]
    constructor
    · rintro ⟨t, ht, rfl⟩; exact ⟨t.isLt, ht⟩
    · rintro ⟨hm, hk⟩; exact ⟨⟨m, hm⟩, hk, rfl⟩
  rw [hmap]
  have hrange : (range (n + 1)).filter (· < k) = range (min (n + 1) k) := by
    ext m
    simp only [mem_filter, mem_range, lt_min_iff]
  rw [hrange, card_range]
  exact Nat.min_comm _ _

/-- **Coverage count (the exchangeability fact, no `sorry`).** Over the uniform
test-index model on `n+1` distinct scores, the number of indices whose score is
covered by the split-conformal set is `min(k, n+1)` — independent of the actual
score values. This is the finite-sample, distribution-free core. -/
theorem coverage_count {n : ℕ} {v : Fin (n + 1) → ℝ} (hv : StrictMono v) (α : ℝ) :
    (univ.filter (fun t => belowCount v t < levelCount n α)).card
      = min (levelCount n α) (n + 1) := by
  have hpred : (univ.filter (fun t : Fin (n + 1) => belowCount v t < levelCount n α))
      = (univ.filter (fun t : Fin (n + 1) => (t : ℕ) < levelCount n α)) := by
    apply filter_congr
    intro t _; rw [belowCount_eq hv t]
  rw [hpred, card_lt_bound]

/-- Marginal coverage of the split-conformal predictor over the uniform
test-index model: the covered fraction. -/
noncomputable def covProb (n : ℕ) (α : ℝ) : ℝ :=
  ((min (levelCount n α) (n + 1) : ℕ) : ℝ) / ((n : ℝ) + 1)

/-- `covProb` is genuinely the covered fraction of the `n+1` test-index model,
for any strictly monotone score family. Ties the definition to the count. -/
theorem covProb_eq_fraction {n : ℕ} {v : Fin (n + 1) → ℝ} (hv : StrictMono v)
    (α : ℝ) :
    covProb n α =
      ((univ.filter (fun t => belowCount v t < levelCount n α)).card : ℝ)
        / ((n : ℝ) + 1) := by
  rw [covProb, ← coverage_count hv]

/-- **Exchangeability ⇒ coverage floor (obligation discharged).** The covered
fraction is at least `⌈(n+1)(1-α)⌉ / (n+1)`. Was a `sorry`; now proven from
`coverage_count` and `levelCount_le`. -/
theorem uniformRankFloor (n : ℕ) (hα : 0 ≤ α) :
    (levelCount n α : ℝ) / ((n : ℝ) + 1) ≤ covProb n α := by
  rw [covProb, min_eq_left (levelCount_le n hα)]

/-- **Split-conformal marginal coverage (C2).** For exchangeable scores the
split-conformal predictor covers at rate `≥ 1 - α`. -/
theorem marginal_coverage (n : ℕ) (hα : 0 ≤ α) :
    (1 - α) ≤ covProb n α :=
  (level_ge n hα).trans (uniformRankFloor n hα)

/-!
## The abstention gate is sound
-/

/-- The gate reports iff the claimed effect meets the minimum detectable effect
at this sample size; otherwise it abstains. -/
def reports (claimedEffect mdes : ℝ) : Prop := mdes ≤ claimedEffect

/-- **Abstention soundness.** If the gate reports, the claimed effect was powered. -/
theorem report_is_powered {claimedEffect mdes : ℝ}
    (h : reports claimedEffect mdes) : mdes ≤ claimedEffect := h

/-- **Underpowered ⇒ abstain.** -/
theorem underpowered_abstains {claimedEffect mdes : ℝ}
    (h : claimedEffect < mdes) : ¬ reports claimedEffect mdes := by
  unfold reports; linarith

end Conformal
