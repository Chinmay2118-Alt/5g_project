# Paper

`main.tex` is the research paper on RAP-FedWMMSE, written on the IEEE conference template (IEEEtran). `main.pdf`
is the compiled final version (Revision 3, 14 pages).

Every number in the paper comes from the saved final results (100 drops over 10 deployments, 95% confidence
intervals), the figure CSVs in [`../rapfedbeam/figures/`](../rapfedbeam/figures/), or the diagnostics in
[`../rapfedbeam/diagnostics/`](../rapfedbeam/diagnostics/). `python -m diagnostics.analysis all`, run from
`rapfedbeam/`, prints every paired statistic quoted in the text (see
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md), section 5.4).

## Compiling

```bash
pdflatex main.tex && pdflatex main.tex
```

To use Overleaf, zip `main.tex` with the `figures/` folder and upload it as a new project (pdfLaTeX).

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper source |
| `main.pdf` | The compiled paper (Revision 3, 14 pages) |
| `figures/` | The 14 result figures (copies of `../rapfedbeam/figures/*.png`) |
| `REVISION_LOG.md` | Google Deep Review revision report: the comments of each round, how each was checked, and what changed |
| `deepreview_round1.md`, `deepreview_round2.md`, `deepreview_round3.md` | The full review texts from Google Deep Review |
