# IEEE paper (final submission, due 20 November 2026, 5 PM)

`main.tex` is the complete research paper, built on the IEEE conference
template (IEEEtran) Sir shared. It compiles locally with MiKTeX
(`pdflatex main.tex`, run twice) to the 13-page `main.pdf` (Revision 2). Every
number in it comes from the saved final results (100 drops over 10
deployments, 95% CIs), the committed figure CSVs in `../rapfedbeam/figures/`,
or the diagnostics in `../rapfedbeam/diagnostics/`.

## Before submitting — check these

1. **Author emails.** The author block lists Kushagra Kumar Arora
   (24BCS061), Chinmay Pramod Ardalkar (24BEC007), Rachit Yadav (24BEC049)
   and Saurav Anand Jha (24BEC054), each as `<rollno>@iiitdwd.ac.in`.
   Verify them, and reorder the authors if the team prefers.
2. Nothing else is a placeholder. Do not change reported numbers by hand;
   if an experiment is rerun, regenerate the figures and update the text
   from the new results.

## Getting it into Overleaf (Sir's Step 1)

1. Go to Overleaf → New Project → **Upload Project**.
2. Upload `Overleaf_Source.zip` (it contains `main.tex` and `figures/`).
3. Set the compiler to pdfLaTeX (default). It compiles as-is.
4. Rename the project as Sir asked (save the IEEE template as your project).

## Deep Review rounds (Sir's Step 2)

- Upload the PDF to Google Deep Review (https://research.google/deepreview/).
- Record every round in `REVISION_LOG.md` (comments received, changes
  made), and export that log to Word or PDF for the final submission.
- At least TWO complete review–revision rounds are mandatory; a third is
  recommended. Rounds 1 and 2 are done; Revision 2 is ready for Round 3.

## Final submission package (Sir's Step 3)

See `SUBMISSION_CHECKLIST.md` for the full list mapped to the Google Form
requirements (100 MB max as ZIP, link if larger).

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper source |
| `main.pdf` | Compiled paper (Revision 2, 13 pages) |
| `figures/` | The 14 result figures (copies of `../rapfedbeam/figures/*.png`) |
| `Overleaf_Source.zip` | Ready-to-upload Overleaf project (not tracked in git; rebuild from `main.tex` and `figures/`) |
| `REVISION_LOG.md` | Deep Review revision report: comments and changes per round |
| `deepreview_round1.md`, `deepreview_round2.md` | The full review texts from Google Deep Review |
| `SUBMISSION_CHECKLIST.md` | Everything the Google Form submission needs |
