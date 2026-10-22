## Likely questions

**Why NMF and not PCA?** PCA gives the best possible fit for a given k, but
its components have positive and negative parts, so a PCA "spectrum" can be
negative at some wavelengths. NMF fits slightly worse (3.18 vs 3.12 on the
toy) but gives parts that can be read as spectra, signatures or concentrations.

**Is the NMF solution unique?** No. Scaling and reordering never change WH,
and overlapping components can blend ("rotational ambiguity"). It becomes
unique only under extra conditions, such as each part having a region
(wavelength or time) where it alone is non-zero. In practice: normalise H, run
many restarts, and check stability.

**How did Alexandrov et al. choose the number of signatures?** In the method
paper (Cell Reports 2013) they bootstrap-resampled the catalogues, ran NMF
many times, and picked the number of signatures where the signatures were
reproducible (high silhouette across runs) and the Frobenius reconstruction
error was low. Our miniature does the same thing with a cosine-based
reproducibility score.

**Is NMF the same as MCR-ALS?** Very close. Multivariate curve resolution with
alternating least squares and non-negativity constraints is the chemometrics
version of the same model. MCR-ALS adds constraints such as closure,
unimodality, or a kinetic model, which can reduce rotational ambiguity.

**Why didn't you fit the kinetic model directly?** You could, if you know the
mechanism. NMF is model-free. It told us there's an intermediate and what it
looks like without assuming A → B → C. The kinetic fit came afterwards, on W.

**What if the data have small negative values (baseline-corrected spectra)?**
NMF needs V ≥ 0. Shift or clip the baseline carefully, model the baseline as
an extra component, or use semi-NMF.

**When should I use KL instead of squared error?** When the noise grows with
the signal, as with counts (sequencing reads, photon counting, mutation
counts). Squared error suits roughly constant Gaussian noise.

**Do the multiplicative updates find the best answer?** They never increase
the misfit (Lee & Seung 2001), but the problem is non-convex, so they can stop
at a local minimum. Lin (2007) discusses convergence in detail. That's why we
use good starts and many restarts.

**Are the signature profiles on screen the real published ones?** No. They're
NMF output from our simulated catalogue, built to follow the published
descriptions (for example, C>T at NpCpG for Signature 1). The published
numbers, such as 4,938,362 mutations and 21 signatures, are quoted from the
paper.
