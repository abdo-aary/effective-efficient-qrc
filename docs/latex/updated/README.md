# QuaRK TMLR manuscript

This directory contains the theory-and-methodology TMLR rewrite of QuaRK, including the readability revision, the finite-family model-selection corollary, and an explicit empirical-question and protocol section. Numerical empirical results remain intentionally deferred; the document is therefore not yet submission-ready.

## TMLR template provenance

The following files are copied without modification from the official
[`JmlrOrg/tmlr-style-file`](https://github.com/JmlrOrg/tmlr-style-file)
repository at commit:

```text
7bf90efe3a0debbba703c05c43f3ff7e4d4a2992
```

| File | SHA-256 |
|---|---|
| `tmlr.sty` | `816214ff5919aa457b6b443bee52b15d9561421417b7f8a50cc84651519f0002` |
| `tmlr.bst` | `306fd454cf40771bee01293eeb98d2c1cd5f4e11ed0cd7296b335f354fc45206` |
| `fancyhdr.sty` | `3d2922548e0e5f1a6c5676eda6ebb6dc20d7d305b4d8c2be5f1c833fb1084e6d` |
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

Do not modify the vendored style files. Update them only by pinning a newer
official commit and recording its checksums here.

## Build

A TeX Live installation with `quantikz`, `latexmk`, and BibTeX is required.
From this directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The default `\usepackage{tmlr}` setting creates the anonymous review version.
Use the official `accepted` or `preprint` package option only for the
corresponding release artifact.

## Manuscript status

- Methodology and theory: rewritten, including the finite-family selection result.
- Empirical evaluation: questions and protocol are explicit, while numerical
  results remain deliberately deferred and visibly marked in
  `sections/experiments.tex`.
- Legacy ACM manuscript: retained separately in `../legacy/` and not used as a
  build dependency.
