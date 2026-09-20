# Integration note, 2026-09-20

Two external neuroscience results and one tooling library to fold into the AIM-DDM decision-phenotype model. These are extractable technique, not authority.

## 1. Superior colliculus action-pressure term (bioRxiv 730072 v2)
SC contributes an additive, side-specific action-promotion term to a logistic/softmax choice model, independent of stimulus evidence. Action item: give the model's choice bias a named neural source by separating an action-pressure current from the evidence-driven drift. This makes the "neurally-grounded parameter profile" claim sharper, since bias (starting point / boundary asymmetry) and evidence (drift) now map to distinct circuits (SC-like action pressure vs cortical/NAcc value) rather than one lumped parameter.

## 2. Neural subspace reorganization in value-based choice (Li et al., iScience 2026, Miller lab)
LPFC population geometry reorganizes across a decision: options in separate order-based subspaces before, chosen vs unchosen rotate orthogonal after, with the chosen subspace expanding and aligning to action. Action item: use this pre/post-decision rotation as an external-validity check that the model's latent decision variable tracks a real population-geometry event, not just behavior. Candidate falsifier: if the fitted decision variable does not correlate with subspace separation on data where both are available, the neural grounding is weaker than claimed.

## 3. CANlab pattern masks (canlab/Neuroimaging_Pattern_Masks)
Validated multivariate fMRI signatures (appetitive/reward, negative affect) as .nii, GPL-3.0 (some patterns need a research agreement). Action item: use a reward signature as an external anchor for the value term instead of training a value decoder from scarce data. Log provenance and licensing before use.

Pairs with the Knutson and Genevsky NAcc reward-anticipation line already in the program.
