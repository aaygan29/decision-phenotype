# NBDT submission checklist

NBDT (Neurons, Behavior, Data analysis, and Theory) is a **diamond open-access, arXiv-overlay**
journal: no submission fee, no APC. You post the manuscript to arXiv first, then submit that link
through NBDT's Scholastica portal (nbdt.scholasticahq.com).

## Files in this folder
- `manuscript.md` / `manuscript.pdf` — the paper (regenerate the PDF with the command below).
- `references.bib` — bibliography.
- `cover_letter.md` — editor cover letter.
- `plain_language_summary.md` — lay summary for circulation.
- `suggested_reviewers.md` — reviewer / feedback candidates.

## Regenerate the PDF
```bash
cd submission && pandoc manuscript.md -o manuscript.pdf --pdf-engine=pdflatex --citeproc
```

## You must fill in / decide before submitting
1. **Affiliation** — add your institution/affiliation under the author name in the YAML front matter
   of `manuscript.md` (currently just "Aayush Gandhi").
2. **ORCID** — 0009-0003-4649-0367 (add on the arXiv + Scholastica forms).
3. **Authorship** — decide whether the `behavioral_decoding` harness lineage (Gowthaam Gopalakrishnan)
   warrants co-authorship or an acknowledgement. If co-author, add to the YAML author list.
4. **Verify DOIs** in `references.bib` (dataset DOIs and newest citations are verified; a few library
   entries are marked "verify before camera-ready").
5. **arXiv category** — primary q-bio.NC (Neurons and Cognition); cross-list cs.LG and stat.ME.
6. **Code release statement** — decide whether to make the `ideal-doodle` repo public at submission or
   on acceptance (the cover letter says "on acceptance"). Note: NBDT does not require double-blind, so
   naming the repo is fine.

## Copy-paste fields (title + abstract) — see below in the response / manuscript front matter.

## Realistic assessment
This is a legitimate NBDT submission: a quantitative, neurally-grounded model of behavior with careful
data analysis and explicit theory about what the data support. Its strengths for NBDT are the honesty
layer, the cross-dataset + individuation results, and the transparent reporting of what is measured vs
simulated vs abstained. Expect reviewers to push on: the underpowered individual neural-to-behavior
link (we abstain, which is the honest answer), the simulation-only status of C5/C6, and the
literature-only fNIRS link. All are disclosed; none is fatal at NBDT.
