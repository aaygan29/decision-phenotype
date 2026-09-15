# Cover letter — NBDT submission

Dear Editors of *Neurons, Behavior, Data analysis, and Theory*,

We submit our manuscript, **"Neural-Grounded Computational Decision Phenotypes:
Honest-by-Construction Prediction of Choice, with Provable Abstention"**, for consideration at NBDT.

**What the paper does.** We build a model that predicts a person's decisions from a small set of
neuroeconomic parameters, grounded in Knutson's Affect-Integration-Motivation account, and we make it
*honest by construction*: every prediction carries a distribution-free coverage guarantee and a
strictly-proper score, and the model abstains when the data cannot support a prediction. We validate
the estimator on real human choices from two public mixed-gambles fMRI datasets, ground the affective
channel in real fMRI and EEG, and — where a direct within-subject neural-to-behavior link is
underpowered — the model abstains and a correlation-transitivity argument shows the grounding does not
require it.

**Why NBDT.** The work is exactly at the intersection NBDT serves: quantitative models of behavior,
grounded in neural data, with careful data analysis and explicit theory about what the data can and
cannot support. Our central methodological stance — reporting simulation results as positive/negative
controls and real-data results as measured effects, and treating abstention as a first-class output —
fits NBDT's emphasis on rigorous, transparent computational neuroscience.

**Openness and reproducibility.** Every number traces to a committed analysis script and a result
file. All datasets are public (NARPS ds001734; Tom et al. 2007 ds000005; EEG ds003458). We are glad to
release the full code repository on acceptance.

**Honesty about scope.** We are explicit throughout about what is established on real data (the
estimator, honesty layer, cross-dataset transfer, individuation, and two-modality neural grounding of
the affective channel) versus what is shown in simulation (the confounded-target abstention and the
attribution-scale extension) versus what remains underpowered and is therefore abstained (the direct
individual neural-to-behavior correlation). No individual is ever scanned, and we make no claim to
measure any specific person's brain.

We believe this is a good fit for NBDT and look forward to the reviewers' feedback.

Sincerely,

Aayush Gandhi
