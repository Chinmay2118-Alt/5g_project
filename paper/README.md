# IEEE paper (final submission, due 20 November 2026, 5 PM)

`main.tex` is the complete research paper, built on the IEEE conference
template (IEEEtran) Sir shared. It compiles locally with MiKTeX
(`pdflatex main.tex`, run twice) to the 11-page `main.pdf` (Revision 1), and every number
in it is taken from the committed figure CSVs in `../rapfedbeam/figures/`
(100 drops over 10 deployments, 95% CIs).

## Before submitting — check these

1. **Author emails.** The author block lists Kushagra Kumar Arora
   (24BCS061), Chinmay Pramod Ardalkar (24BEC007), Rachit Yadav (24BEC049)
   and Saurav Anand Jha (24BEC054). The teammate emails use the institute
   pattern `<rollno>@iiitdwd.ac.in` — verify they are right, and reorder
   the authors if the team prefers a different order.
2. Nothing else is a placeholder. Do not change reported numbers by hand;
   if an experiment is rerun, regenerate the figures and update the text
   from the new CSVs.

## Getting it into Overleaf (Sir's Step 1)

1. Go to Overleaf → New Project → **Upload Project**.
2. Upload `Overleaf_Source.zip` (it contains `main.tex` and `figures/`).
3. Set the compiler to pdfLaTeX (default). It compiles as-is.
4. Rename the project as Sir asked (save the IEEE template as your project).

## Deep Review rounds (Sir's Step 2)

- Download the PDF from Overleaf and upload it to Google Deep Review
  (https://research.google/deepreview/).
- Record every round in `REVISION_LOG.md` (comments received, changes
  made), and export that log to Word or PDF for the final submission.
- At least TWO complete review–revision rounds are mandatory; a third is
  recommended.

## Final submission package (Sir's Step 3)

See `SUBMISSION_CHECKLIST.md` for the full list mapped to the Google Form
requirements (100 MB max as ZIP, link if larger).

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper source |
| `main.pdf` | Compiled paper (Revision 1, 11 pages) |
| `figures/` | The 14 result figures (copies of `../rapfedbeam/figures/*.png`) |
| `Overleaf_Source.zip` | Ready-to-upload Overleaf project |
| `REVISION_LOG.md` | Deep Review revision report: comments and changes per round (Round 1 done) |
| `deepreview_round1.md` | The full Round 1 review text from Google Deep Review |
| `SUBMISSION_CHECKLIST.md` | Everything the Google Form submission needs |
