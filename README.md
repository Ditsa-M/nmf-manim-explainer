# NMF, explained — a 15-minute class presentation

An animated presentation of **Non-negative Matrix Factorization (NMF)** for a
biochemistry transform-methods assignment, built with
[Manim](https://www.manim.community/). It is organised around the four
questions the class should be able to answer: what goes in, what comes out,
what is kept or lost, and which biochemical question gets easier. It runs
14:24, under the 15-minute limit. The narration is **burned in as captions**
and split between two speakers with equal time (A 6:55, B 7:10). The level
rises from **beginner to graduate**; a badge in the corner shows the level,
and on-screen citations point to the papers. The video ends with a reading
list in that same order. The maths is
typeset with LaTeX, and the biochemistry is illustrated with drawn pictures:
a spectrometer, a stopped-flow mixer, DNA, a tumour cell, and icons for
mutation causes.

▶ **Watch:** [`NMF_explained.mp4`](https://github.com/Ditsa-M/nmf-manim-explainer/releases/latest/download/NMF_explained.mp4)
(1080p60, 14:24, captions burned in; attached to the [latest release](https://github.com/Ditsa-M/nmf-manim-explainer/releases/latest),
with soft subtitles in [`NMF_explained.srt`](NMF_explained.srt))
· 🎙 **Two-speaker script with timestamps, rubric checklist and Q&A prep:** [`SCRIPT.md`](SCRIPT.md)

Every number on screen is computed in [`helpers.py`](helpers.py) with
numpy/scikit-learn. The only exceptions are the published counts quoted from
Alexandrov et al. (2013), which are kept as named, cited constants.

## What the video covers

| Part | Scene class | Topic | Speaker |
|---|---|---|---|
| 1 · How it works | `S01Intro` | Spectrometer, overlapping spectra; the four questions | A |
| | `S02Input` | V ∈ ℝ≥0^(n×m); Beer–Lambert A = ℓCE is already an NMF | A |
| | `S03Output` | V ≈ WH; (WH)ᵢⱼ = Σᵣ WᵢᵣHᵣⱼ; sum of rank-one layers; rank ≤ k, compression | A |
| | `S04Fitting` | Frobenius / KL ↔ Gaussian / Poisson likelihood; gradient → multiplicative updates; KKT; non-convexity | A |
| | `S05KeptLost` | Geometry: NMF cone vs PCA axes; kept vs lost; Eckart–Young | B |
| 2 · Biochemical use | `S06Kinetics` | Before/after: simulated A → B → C stopped-flow spectra; intermediate spectrum and rate constants | B |
| | `S07Knobs` | Researcher's choices: k, β-divergence, initialization, regularization, preprocessing, solver | B |
| | `S08Questions` | Biochemical questions NMF makes easier | B |
| 3 · Literature | `S09Paper` | Alexandrov et al. 2013 (Nature): tumour → DNA → 96-class catalogue | A |
| | `S10Method` | M ≈ PE, bootstrap, choosing k; reproduced in miniature with cosine similarity | A |
| | `S11KeyFigure` | Fig. 2 signatures and their causes (age, APOBEC, smoking, UV); Fig. 3-style map from W; insight | B |
| 4 · Strengths & limits | `S12StrengthsLimits` | Strengths (before/after); four failure modes, each with a computed figure | A, B |
| | `S13Theory` | Graduate level: (WT)(T⁻¹H) ambiguity, simplex geometry of uniqueness, NP-hardness, separability, solvers | B |
| | `S14Conclusions` | Conclusions *not* justified; the four questions; reading list beginner → graduate | A, B |

Exact timestamps for every line are in [`SCRIPT.md`](SCRIPT.md).

Conventions follow scikit-learn: V is *samples × features*, **W** (samples × k)
holds the **amounts** and **H** (k × features) holds the **parts**.

All examples are **synthetic**: the 30 mixture spectra, the kinetics experiment
and the 120-genome mutation catalogue. The signature profiles in `S11` are NMF
output from the simulated catalogue, shaped after the paper's descriptions.
They are not the published profiles, and the video says so on screen.

## Setup

### Option A: pip (normal machine)

Needs Python 3.10–3.12, plus cairo, pango and ffmpeg from your system package manager.

```bash
# Ubuntu/Debian:
sudo apt install libcairo2-dev libpango1.0-dev ffmpeg
# macOS:  brew install cairo pango ffmpeg pkg-config

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

No `sudo` and no `python3-venv`? [uv](https://docs.astral.sh/uv/) works too:
`uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`.

### Option B: micromamba / conda (no sudo, no system libraries)

conda-forge ships cairo, pango and ffmpeg itself, so this works on a locked-down machine:

```bash
# one-off install of micromamba into ./.mamba (no root needed)
export MAMBA_ROOT_PREFIX=$PWD/.mamba/root
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C .mamba bin/micromamba
.mamba/bin/micromamba create -y -p ./.env -f environment.yml
```

### LaTeX (needed for the maths)

The equations use Manim's `MathTex`, which needs `latex` and `dvisvgm`.
With sudo, run `sudo apt install texlive texlive-latex-extra dvisvgm`
(macOS: MacTeX). Without sudo, a ~175 MB user-level TeX Live works. The render
scripts pick it up automatically from `~/texlive/bin/*/`:

```bash
curl -LO https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar xzf install-tl-unx.tar.gz && cd install-tl-2*/
printf '%s\n' "selected_scheme scheme-basic" "TEXDIR $HOME/texlive" \
  "TEXMFLOCAL $HOME/texlive/texmf-local" "TEXMFSYSVAR $HOME/texlive/texmf-var" \
  "TEXMFSYSCONFIG $HOME/texlive/texmf-config" "TEXMFHOME $HOME/texmf" \
  "tlpdbopt_install_docfiles 0" "tlpdbopt_install_srcfiles 0" "instopt_adjustpath 0" > tl.profile
./install-tl --profile=tl.profile
~/texlive/bin/*/tlmgr install standalone preview dvisvgm amsfonts babel-english
```

## Rendering

```bash
./render.sh S06Kinetics -ql    # one scene, quick 480p preview
./render.sh S06Kinetics        # one scene, 1080p60
./render_all.sh -ql            # all scenes at 480p → NMF_explained_480p15.mp4
./render_all.sh                # all scenes at 1080p60 → NMF_explained.mp4
```

`render_all.sh` renders all scenes in parallel and concatenates them in order
with ffmpeg. It then runs `make_script.py`, which turns the captions each scene
logged into `NMF_explained.srt` and `SCRIPT.md`. Finally it prints each scene's
duration and the total, checked with ffprobe. The scripts use `.venv/` or
`.env/` automatically if either exists. On a 20-core machine the full 1080p
render takes about ten minutes.

Scenes are paced by their narration. `NMFScene.say(text, *animations)` shows
the text as a caption, plays the animations, and holds for as long as the text
takes to read aloud at 150 wpm. The narration therefore lives in one place,
`nmf.py`; edit it there and re-render, and the captions, `.srt` and
`SCRIPT.md` all follow.

## Repository layout

```
.
├── nmf.py            # the 13 Manim scenes, in four parts, with their narration
├── helpers.py        # all data + numbers (toy V, NMF fits, MU iterations,
│                     #   synthetic spectra, kinetics, mutation catalogue,
│                     #   k-sweeps, init comparison, 2-D geometry),
│                     #   colours, TextMatrix, axes helpers, captioned NMFScene base
├── pictures.py       # drawn illustrations (spectrometer, stopped-flow, DNA, icons)
├── render.sh         # render one scene
├── render_all.sh     # render all + concatenate → NMF_explained.mp4 (+ .srt, SCRIPT.md)
├── make_script.py    # captions → NMF_explained.srt + SCRIPT.md
├── script_qa.md      # hand-written Q&A prep, appended to SCRIPT.md
├── SCRIPT.md         # generated: two-speaker narration, timestamps, rubric map, Q&A
├── NMF_explained.srt # generated: soft subtitles
├── requirements.txt  # pip route
├── environment.yml   # conda/micromamba route
└── LICENSE           # MIT
```

## References (beginner → graduate)

**Beginner**
1. D. D. Lee & H. S. Seung, "Learning the parts of objects by non-negative matrix factorization," *Nature* **401**, 788–791 (1999). [doi:10.1038/44565](https://doi.org/10.1038/44565)
2. N. Gillis, "The why and how of nonnegative matrix factorization," in *Regularization, Optimization, Kernels, and Support Vector Machines*, pp. 275–310 (Chapman & Hall/CRC, 2014). [doi:10.1201/b17558-15](https://doi.org/10.1201/b17558-15)
3. J.-P. Brunet, P. Tamayo, T. R. Golub & J. P. Mesirov, "Metagenes and molecular pattern discovery using matrix factorization," *PNAS* **101**(12), 4164–4169 (2004). [doi:10.1073/pnas.0308531101](https://doi.org/10.1073/pnas.0308531101)

**Intermediate: origins, algorithms and applications**

4. P. Paatero & U. Tapper, "Positive matrix factorization: A non-negative factor model with optimal utilization of error estimates of data values," *Environmetrics* **5**(2), 111–126 (1994). [doi:10.1002/env.3170050203](https://doi.org/10.1002/env.3170050203)
5. D. D. Lee & H. S. Seung, "Algorithms for non-negative matrix factorization," *Advances in Neural Information Processing Systems 13* (NIPS 2000, published 2001). [proceedings](https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html)
6. A. de Juan & R. Tauler, "Multivariate curve resolution (MCR) from 2000: progress in concepts and applications," *Crit. Rev. Anal. Chem.* **36**(3–4), 163–176 (2006). [doi:10.1080/10408340600970005](https://doi.org/10.1080/10408340600970005)
7. L. B. Alexandrov et al., "Signatures of mutational processes in human cancer," *Nature* **500**, 415–421 (2013). [doi:10.1038/nature12477](https://doi.org/10.1038/nature12477), **the literature example** (Part 3)
   - Method paper: L. B. Alexandrov, S. Nik-Zainal, D. C. Wedge, P. J. Campbell & M. R. Stratton, "Deciphering signatures of mutational processes operative in human cancer," *Cell Reports* **3**(1), 246–259 (2013). [doi:10.1016/j.celrep.2012.12.008](https://doi.org/10.1016/j.celrep.2012.12.008)
8. D. Kotliar et al., "Identifying gene expression programs of cell-type identity and cellular activity with single-cell RNA-Seq," *eLife* **8**, e43803 (2019). [doi:10.7554/eLife.43803](https://doi.org/10.7554/eLife.43803)
9. E. A. Pnevmatikakis et al., "Simultaneous denoising, deconvolution, and demixing of calcium imaging data," *Neuron* **89**(2), 285–299 (2016). [doi:10.1016/j.neuron.2015.11.037](https://doi.org/10.1016/j.neuron.2015.11.037)

**Advanced: practice and solvers**

10. C. Boutsidis & E. Gallopoulos, "SVD based initialization: A head start for nonnegative matrix factorization," *Pattern Recognition* **41**(4), 1350–1362 (2008). [doi:10.1016/j.patcog.2007.09.010](https://doi.org/10.1016/j.patcog.2007.09.010)
11. P. O. Hoyer, "Non-negative matrix factorization with sparseness constraints," *JMLR* **5**, 1457–1469 (2004). [jmlr.org](https://www.jmlr.org/papers/v5/hoyer04a.html)
12. C.-J. Lin, "Projected gradient methods for nonnegative matrix factorization," *Neural Computation* **19**(10), 2756–2779 (2007). [doi:10.1162/neco.2007.19.10.2756](https://doi.org/10.1162/neco.2007.19.10.2756)
13. C.-J. Lin, "On the convergence of multiplicative update algorithms for nonnegative matrix factorization," *IEEE Trans. Neural Networks* **18**(6), 1589–1596 (2007). [doi:10.1109/TNN.2007.895831](https://doi.org/10.1109/TNN.2007.895831)
14. H. Kim & H. Park, "Nonnegative matrix factorization based on alternating nonnegativity constrained least squares and active set method," *SIAM J. Matrix Anal. Appl.* **30**(2), 713–730 (2008). [doi:10.1137/07069239X](https://doi.org/10.1137/07069239X)
15. A. Cichocki & A.-H. Phan, "Fast local algorithms for large scale nonnegative matrix and tensor factorizations," *IEICE Trans. Fundamentals* **E92-A**(3), 708–721 (2009). [doi:10.1587/transfun.E92.A.708](https://doi.org/10.1587/transfun.E92.A.708)
16. C. Févotte & J. Idier, "Algorithms for nonnegative matrix factorization with the β-divergence," *Neural Computation* **23**(9), 2421–2456 (2011). [doi:10.1162/NECO_a_00168](https://doi.org/10.1162/NECO_a_00168)

**Graduate: identifiability and complexity**

17. D. Donoho & V. Stodden, "When does non-negative matrix factorization give a correct decomposition into parts?" *Advances in Neural Information Processing Systems 16* (2003). [proceedings](https://papers.nips.cc/paper_files/paper/2003/hash/1843e35d41ccf6e63273495ba42df3c1-Abstract.html)
18. S. A. Vavasis, "On the complexity of nonnegative matrix factorization," *SIAM J. Optim.* **20**(3), 1364–1377 (2009). [doi:10.1137/070709967](https://doi.org/10.1137/070709967)
19. S. Arora, R. Ge, R. Kannan & A. Moitra, "Computing a nonnegative matrix factorization — provably," *Proc. 44th ACM STOC*, 145–162 (2012). [doi:10.1145/2213977.2213994](https://doi.org/10.1145/2213977.2213994)
20. K. Huang, N. D. Sidiropoulos & A. Swami, "Non-negative matrix factorization revisited: uniqueness and algorithm for symmetric decomposition," *IEEE Trans. Signal Process.* **62**(1), 211–224 (2014). [doi:10.1109/TSP.2013.2285514](https://doi.org/10.1109/TSP.2013.2285514)
21. X. Fu, K. Huang, N. D. Sidiropoulos & W.-K. Ma, "Nonnegative matrix factorization for signal and data analytics: identifiability, algorithms, and applications," *IEEE Signal Process. Mag.* **36**(2), 59–80 (2019). [doi:10.1109/MSP.2018.2877582](https://doi.org/10.1109/MSP.2018.2877582)
22. N. Gillis, *Nonnegative Matrix Factorization* (SIAM, 2020). [doi:10.1137/1.9781611976410](https://doi.org/10.1137/1.9781611976410)

All titles, journals, volumes, pages and DOIs above were checked against
Crossref, or the NeurIPS and JMLR sites for refs. 5, 11 and 17. Not checked:
the page numbers of ref. 5 (usually cited as pp. 556–562) and ref. 17, so they
are left out.

## Notes and caveats

- All data are **synthetic**: Gaussian bands with Gaussian noise, and Poisson
  mutation counts. That's why the recovery is so clean. Real data have
  baselines, scattering and model error, as Part 4 points out.
- The numbers quoted from Alexandrov et al. (2013), and the signature
  aetiologies, were checked against the full text on PubMed Central
  (PMC3776390). The NMF, bootstrap and k-selection details were checked
  against the Cell Reports methods paper (PMC3588146).
- The "NNDSVDa" comparison uses the plain multiplicative-update loop from
  `helpers.mu_run`, not scikit-learn's default coordinate-descent solver.
- `helpers.py` imports `sklearn.decomposition._nmf._initialize_nmf`, a private
  function. It was tested with scikit-learn 1.9.1, Manim 0.22.0, numpy 2.5.4 and
  scipy 1.18.1.

## License

MIT — see [LICENSE](LICENSE).
