"""Shared data, colours and drawing helpers for the NMF explainer.

Every number that appears on screen is computed here (numpy / scikit-learn),
so the scenes never contain hand-typed values.

Convention (same as scikit-learn): V is samples x features, V ≈ W H with
  W = samples x k   ("amounts": how much of each part a sample contains)
  H = k x features  ("parts": what each part looks like across the features)

LaTeX is not assumed, so matrices are drawn with Text + Line brackets.
"""
import json
import os
import re
import textwrap
import warnings

import numpy as np
from manim import (BLACK, BLUE, DOWN, GREEN, GREY_B, LEFT, ORANGE, RIGHT, UL, UP, UR, WHITE, Axes,
                   FadeIn, FadeOut, MathTex, Scene, Text, VGroup, VMobject, config)
from scipy.optimize import least_squares, linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _initialize_nmf

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# ---------------------------------------------------------------- colours
C1, C2, C3 = ORANGE, BLUE, GREEN          # component 1, 2, 3 (everywhere)
COMP_COLORS = [C1, C2, C3]
FONT = "DejaVu Sans"                       # covers ‖ ⊙ ⊘ ᵀ ᵢ ⱼ ² ≥ ≈ Σ
EPS = 1e-12


def T(text, size=32, color=WHITE, **kw):
    """Text in the project font."""
    return Text(text, font=FONT, font_size=size, color=color, **kw)


def M(tex, size=38, color=WHITE, **kw):
    """LaTeX maths (needs latex + dvisvgm on PATH)."""
    return MathTex(tex, font_size=size, color=color, **kw)


def fit_box(mob, width, height):
    """Shrink `mob` (never enlarge) to fit inside width x height."""
    f = min(1.0, width / max(mob.width, 1e-6), height / max(mob.height, 1e-6))
    return mob.scale(f)


# ------------------------------------------------------------ toy example
V_DATA = np.array([[5, 3, 0], [4, 0, 0], [1, 1, 5], [0, 1, 4]], dtype=float)
_toy = NMF(n_components=2, init="nndsvd", max_iter=5000, random_state=0)
W_FULL = _toy.fit_transform(V_DATA)
H_FULL = _toy.components_
W_DATA = np.round(W_FULL, 1)               # what is displayed
H_DATA = np.round(H_FULL, 1)
WH_DATA = W_DATA @ H_DATA                  # reconstruction from displayed numbers
RESID = V_DATA - WH_DATA
FRO2 = float(np.sum(RESID ** 2))           # ‖V − WH‖²_F with displayed W, H


def kl_div(V, R):
    """Generalised KL divergence D(V‖R) (0·log0 = 0)."""
    R = np.maximum(R, EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(V > 0, V * np.log(V / R), 0.0)
    return float(np.sum(t - V + R))


KL_OF_FRO_FIT = kl_div(V_DATA, WH_DATA)
_toy_kl = NMF(n_components=2, init="nndsvda", solver="mu",
              beta_loss="kullback-leibler", max_iter=5000, random_state=0)
_Wk = _toy_kl.fit_transform(V_DATA)
KL_OF_KL_FIT = kl_div(V_DATA, _Wk @ _toy_kl.components_)

# rank-2 SVD of the toy matrix (for the "why non-negative" comparison)
_U, _S, _Vt = np.linalg.svd(V_DATA, full_matrices=False)
SVD_COMPS = _Vt[:2]                        # 2 x 3, mixed signs
_SVD_SIGN = np.sign(SVD_COMPS.sum(axis=1, keepdims=True))
SVD_COMPS = SVD_COMPS * _SVD_SIGN          # make each mostly positive
SVD_RECON = (_U[:, :2] * _S[:2]) @ _Vt[:2]
SVD_FRO2 = float(np.sum((V_DATA - SVD_RECON) ** 2))
NMF_FULL_FRO2 = float(np.sum((V_DATA - W_FULL @ H_FULL) ** 2))


# ----------------------------------------------- multiplicative updates
def mu_run(V, W, H, n_iter):
    """Lee & Seung multiplicative updates for ½‖V − WH‖²_F.

    Returns lists of W, H and loss (½‖·‖²) after 0..n_iter iterations.
    """
    Ws, Hs, losses = [W.copy()], [H.copy()], [0.5 * np.sum((V - W @ H) ** 2)]
    for _ in range(n_iter):
        H = H * (W.T @ V) / (W.T @ W @ H + EPS)
        W = W * (V @ H.T) / (W @ H @ H.T + EPS)
        Ws.append(W.copy())
        Hs.append(H.copy())
        losses.append(0.5 * np.sum((V - W @ H) ** 2))
    return Ws, Hs, np.array(losses)


_rng = np.random.default_rng(3)
MU_W0 = np.round(_rng.uniform(0.2, 2.0, size=(4, 2)), 1)
MU_H0 = np.round(_rng.uniform(0.2, 2.0, size=(2, 3)), 1)
MU_N = 200
MU_W, MU_H, MU_LOSS = mu_run(V_DATA, MU_W0, MU_H0, MU_N)
assert np.all(np.diff(MU_LOSS) <= 1e-9), "MU loss should be non-increasing"
# the four quantities in the first H update, for the on-screen walk-through
MU_WtV = MU_W0.T @ V_DATA
MU_WtWH = MU_W0.T @ MU_W0 @ MU_H0
MU_H1 = MU_H0 * MU_WtV / MU_WtWH

# --------------------------------------------- synthetic spectral mixture
WL = np.arange(400, 701, 1.0)              # wavelength, nm


def _band(mu, sig, amp):
    return amp * np.exp(-0.5 * ((WL - mu) / sig) ** 2)


S_TRUE = np.vstack([
    _band(455, 18, 1.0) + _band(520, 14, 0.35),     # component 1 (orange)
    _band(505, 24, 1.0),                            # component 2 (blue)
    _band(580, 20, 0.9) + _band(630, 26, 0.45),     # component 3 (green)
])
S_TRUE = S_TRUE / S_TRUE.max(axis=1, keepdims=True)
N_SAMPLES = 30
_srng = np.random.default_rng(7)
C_TRUE = _srng.gamma(1.5, 0.5, size=(N_SAMPLES, 3))
NOISE_SD = 0.02
X_SPEC = np.clip(C_TRUE @ S_TRUE + _srng.normal(0, NOISE_SD, (N_SAMPLES, WL.size)), 0, None)

_spec = NMF(n_components=3, init="nndsvda", max_iter=20000, tol=1e-7, random_state=0)
_W_spec = _spec.fit_transform(X_SPEC)
_H_spec = _spec.components_


def match_components(H_est, H_ref):
    """Permutation of H_est rows that best matches H_ref (by correlation)."""
    k = H_ref.shape[0]
    corr = np.corrcoef(H_est, H_ref)[:k, k:] if H_est.shape[0] == k else \
        np.corrcoef(np.vstack([H_est, H_ref]))[:H_est.shape[0], H_est.shape[0]:]
    rows, cols = linear_sum_assignment(-corr)
    perm = np.empty(k, dtype=int)
    perm[cols] = rows
    return perm, corr[perm, np.arange(k)]


_perm, SPEC_CORR = match_components(_H_spec, S_TRUE)
_scale = _H_spec[_perm].max(axis=1)
S_EST = _H_spec[_perm] / _scale[:, None]                # peak-normalised
C_EST = _W_spec[:, _perm] * _scale[None, :]             # undo the scaling
CONC_CORR = np.array([np.corrcoef(C_EST[:, i], C_TRUE[:, i])[0, 1] for i in range(3)])
SPEC_RESID_RMS = float(np.sqrt(np.mean((X_SPEC - _W_spec @ _H_spec) ** 2)))
SPEC_COMPRESS = (X_SPEC.size, 3 * (X_SPEC.shape[0] + X_SPEC.shape[1]))

# ---------------------------------------------- practical issues (spectra)
INIT_ITERS = 300
# NNDSVDa (zeros filled with the data mean): plain NNDSVD leaves exact zeros,
# which multiplicative updates can never move away from.
_W_nd, _H_nd = _initialize_nmf(X_SPEC, 3, init="nndsvda")
INIT_NNDSVD = mu_run(X_SPEC, _W_nd, _H_nd, INIT_ITERS)[2]
INIT_RANDOM = []
for seed in range(6):
    Wr, Hr = _initialize_nmf(X_SPEC, 3, init="random", random_state=seed)
    INIT_RANDOM.append(mu_run(X_SPEC, Wr, Hr, INIT_ITERS)[2])
INIT_RANDOM = np.array(INIT_RANDOM)

# non-uniqueness: W H = (W D)(D⁻¹ H)
D_DIAG = np.array([2.0, 0.5])
W_SCALED = W_DATA * D_DIAG[None, :]
H_SCALED = H_DATA / D_DIAG[:, None]
SCALE_MAXDIFF = float(np.max(np.abs(W_SCALED @ H_SCALED - WH_DATA)))

# choosing k: reconstruction error and stability across restarts
K_RANGE = list(range(1, 7))
K_ERR, K_STAB = [], []
for k in K_RANGE:
    fits = []
    for seed in range(8):
        m = NMF(n_components=k, init="random", max_iter=3000, tol=1e-6, random_state=seed)
        m.fit(X_SPEC)
        fits.append((m.reconstruction_err_, m.components_))
    fits.sort(key=lambda f: f[0])
    K_ERR.append(fits[0][0])
    ref = fits[0][1]
    if k == 1:
        K_STAB.append(1.0)
        continue
    scores = []
    for _, Hk in fits[1:]:
        _, c = match_components(Hk, ref)
        scores.append(c.mean())
    K_STAB.append(float(np.mean(scores)))
K_ERR = np.array(K_ERR)
K_STAB = np.array(K_STAB)

# sparsity: L1 penalty on W
_SP_ALPHA = 0.005


def _zero_frac(alpha):
    m = NMF(n_components=3, init="nndsvda", alpha_W=alpha, alpha_H="same",
            l1_ratio=1.0, max_iter=20000, tol=1e-7, random_state=0)
    Wm = m.fit_transform(X_SPEC)
    return (float(np.mean(m.components_ < 1e-6)), float(np.mean(Wm < 1e-6)),
            float(m.reconstruction_err_))


SPARSE_ALPHA = _SP_ALPHA
SPARSE_H0, SPARSE_W0, SPARSE_ERR0 = _zero_frac(0.0)
SPARSE_H1, SPARSE_W1, SPARSE_ERR1 = _zero_frac(_SP_ALPHA)


# ------------------------------------------- use case: reaction kinetics
# Synthetic stopped-flow UV-vis experiment, A -> B -> C (first order).
# V = time points x wavelengths; NMF (k = 3) should give the three pure
# spectra (H) and the three concentration time courses (W).
KIN_T = np.linspace(0, 100, 41)            # s
KIN_K1, KIN_K2 = 0.2, 0.05                 # s⁻¹, true rate constants


def kin_conc(t, k1, k2):
    """[A], [B], [C] for A -> B -> C with [A]0 = 1."""
    A = np.exp(-k1 * t)
    B = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    return np.c_[A, B, 1 - A - B]


KIN_C = kin_conc(KIN_T, KIN_K1, KIN_K2)
KIN_S = np.vstack([_band(430, 20, 1.0),                         # A
                   _band(515, 18, 1.0),                         # B (intermediate)
                   _band(600, 25, 1.0) + _band(560, 20, 0.3)])  # C
KIN_S = KIN_S / KIN_S.max(axis=1, keepdims=True)
KIN_NOISE = 0.01
_krng = np.random.default_rng(11)
KIN_X = np.clip(KIN_C @ KIN_S + _krng.normal(0, KIN_NOISE, (KIN_T.size, WL.size)), 0, None)
_kin = NMF(n_components=3, init="nndsvda", max_iter=50000, tol=1e-10, random_state=0)
_W_kin = _kin.fit_transform(KIN_X)
_kperm, KIN_SPEC_CORR = match_components(_kin.components_, KIN_S)
_kscale = _kin.components_[_kperm].max(axis=1)
KIN_S_EST = _kin.components_[_kperm] / _kscale[:, None]        # peak-normalised spectra
KIN_W_EST = _W_kin[:, _kperm] * _kscale[None, :]               # matching amounts


def _kin_resid(p):
    k1, k2, a0, a1, a2 = p
    return (KIN_W_EST - kin_conc(KIN_T, k1, k2) * np.array([a0, a1, a2])).ravel()


# global kinetic fit to the NMF time courses (amplitudes free: W has no absolute scale)
_kfit = least_squares(_kin_resid, [0.1, 0.02, 1, 1, 1],
                      bounds=([1e-3, 1e-3, 0, 0, 0], [2, 2, 10, 10, 10]))
KIN_K1_EST, KIN_K2_EST = _kfit.x[:2]
KIN_AMP = _kfit.x[2:]
KIN_B_PEAK = float(WL[np.argmax(KIN_S_EST[1])])
KIN_COMPRESS = (KIN_X.size, 3 * sum(KIN_X.shape))

# ------------------------------------------- literature: published numbers
# Alexandrov et al., Nature 500, 415–421 (2013), doi:10.1038/nature12477
# (Results/Methods). These are the only numbers typed in by hand.
PAPER_MUTATIONS = 4_938_362
PAPER_CANCERS = 7_042
PAPER_GENOMES, PAPER_EXOMES = 507, 6_535
PAPER_TYPES = 30
PAPER_SIGNATURES = 21

# -------------------------------- literature: mutational signatures in miniature
# Alexandrov et al. (2013): each genome -> counts in 96 classes
# (6 substitutions x 16 flanking-base contexts). Here: 4 schematic signatures
# shaped after the published descriptions, 120 simulated genomes, Poisson counts.
SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
SUB_COLORS = ["#03BCEE", "#FFFFFF", "#E32926", "#CAC9C9", "#A1CE63", "#EBC6C4"]
_BASES = "ACGT"
MUT_CHANNELS = [(s, f, g) for s in SUBS for f in _BASES for g in _BASES]


def _sig(rule):
    v = np.array([rule(s, f, g) for s, f, g in MUT_CHANNELS], dtype=float) + 0.01
    return v / v.sum()


MUT_SIG_NAMES = ["Signature 1", "Signatures 2 + 13", "Signature 4", "Signature 7"]
MUT_SIG = np.vstack([
    _sig(lambda s, f, g: 1.0 if (s == "C>T" and g == "G") else 0),               # C>T at NpCpG
    _sig(lambda s, f, g: (1.0 if g in "AT" else 0.35) if (s == "C>T" and f == "T")
         else (0.7 if g in "AT" else 0.25) if (s == "C>G" and f == "T") else 0),  # TpCpN
    _sig(lambda s, f, g: 1.0 if s == "C>A" else 0.25 if s == "C>T"
         else 0.15 if s == "T>A" else 0),                                         # mostly C>A
    _sig(lambda s, f, g: 1.0 if (s == "C>T" and f in "CT") else 0.1 if s == "C>T" else 0),  # C>T after a pyrimidine
])
MUT_G = 120
# four simulated groups of 30 genomes ("tissue types"), each with its own
# chance of carrying signatures 2+13, 4 and 7; signature 1 is in every genome
MUT_GROUP_NAMES = ["group 1", "group 2", "group 3", "group 4"]
MUT_GROUP = np.repeat(np.arange(4), MUT_G // 4)
_P_ON = np.array([[0.15, 0.15, 0.90],                  # UV-dominated
                  [0.15, 0.90, 0.15],                  # smoking-dominated
                  [0.90, 0.15, 0.15],                  # APOBEC-dominated
                  [0.10, 0.10, 0.10]])                 # mostly background
_mrng = np.random.default_rng(5)
_E = np.zeros((MUT_G, 4))
_E[:, 0] = _mrng.gamma(2, 1, MUT_G)
for _j in (1, 2, 3):
    _on = _mrng.random(MUT_G) < _P_ON[MUT_GROUP, _j - 1]
    _E[_on, _j] = _mrng.gamma(2, 2, _on.sum())
_tot = np.exp(_mrng.normal(6.5, 0.6, MUT_G))
MUT_E = _E / _E.sum(axis=1, keepdims=True) * _tot[:, None]
MUT_V = _mrng.poisson(MUT_E @ MUT_SIG).astype(float)   # genomes x 96 counts
MUT_TOTAL = int(MUT_V.sum())


def _unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def mut_fit(k, seed, exposures=False):
    m = NMF(n_components=k, init="random", solver="mu", beta_loss="kullback-leibler",
            max_iter=5000, tol=1e-7, random_state=seed)
    W = m.fit_transform(MUT_V)
    tot = m.components_.sum(axis=1, keepdims=True)
    if exposures:                                      # mutations attributed to each process
        return m.reconstruction_err_, m.components_ / tot, W * tot.T
    return m.reconstruction_err_, m.components_ / tot


def _cos_match(Ha, Hb):
    cos = _unit(Ha) @ _unit(Hb).T
    r, c = linear_sum_assignment(-cos)
    perm = np.empty(Hb.shape[0], dtype=int)
    perm[c] = r
    return perm, cos[perm, np.arange(Hb.shape[0])]


# choose k as Alexandrov et al. did: reproducible across restarts + low error
MUT_K_RANGE = list(range(2, 8))
MUT_K_ERR, MUT_K_STAB = [], []
_mut_best, _mut_seed = {}, {}
for _k in MUT_K_RANGE:
    _fits = sorted(((mut_fit(_k, sd), sd) for sd in range(6)), key=lambda f: f[0][0])
    _mut_seed[_k] = _fits[0][1]
    _fits = [f for f, _ in _fits]
    _mut_best[_k] = _fits[0][1]
    MUT_K_ERR.append(_fits[0][0])
    MUT_K_STAB.append(float(np.mean([_cos_match(Hf, _fits[0][1])[1].min() for _, Hf in _fits[1:]])))
MUT_K_ERR, MUT_K_STAB = np.array(MUT_K_ERR), np.array(MUT_K_STAB)
_mperm, MUT_COS = _cos_match(_mut_best[4], MUT_SIG)
MUT_SIG_EST = _mut_best[4][_mperm]
# Fig. 3-style summary from the NMF exposures: in each group, the fraction of
# genomes where a signature explains > 5 % of mutations, and its median share
_, _, _E_est = mut_fit(4, _mut_seed[4], exposures=True)
_share = _E_est[:, _mperm] / _E_est.sum(axis=1, keepdims=True)
FIG3_FRAC = np.array([[np.mean(_share[MUT_GROUP == g, r] > 0.05) for r in range(4)] for g in range(4)])
FIG3_MED = np.array([[np.median(_share[MUT_GROUP == g, r]) for r in range(4)] for g in range(4)])

# ------------------------------------------- failure-mode figures
# (a) rotational ambiguity: two broad, overlapping bands and no pure samples;
#     NMF finds different spectra that fit the data just as well as the truth
ROT_S = np.vstack([_band(500, 45, 1.0), _band(560, 45, 1.0)])
ROT_X = np.random.default_rng(4).uniform(0.3, 1.0, (40, 2)) @ ROT_S
_rot = NMF(n_components=2, init="random", random_state=0, max_iter=20000, tol=1e-10)
_rot.fit(ROT_X)
_ro = np.argsort(np.argmax(_rot.components_, axis=1))
ROT_H = (_rot.components_ / _rot.components_.max(axis=1, keepdims=True))[_ro]
ROT_PEAKS_TRUE = WL[np.argmax(ROT_S, axis=1)]
ROT_PEAKS_NMF = WL[np.argmax(ROT_H, axis=1)]
ROT_REL_ERR = float(_rot.reconstruction_err_ / np.linalg.norm(ROT_X))     # truth: exactly 0

# (b) k too large: k = 4 on the 3-species spectra
_k4 = NMF(n_components=4, init="nndsvda", max_iter=20000, tol=1e-9, random_state=0).fit(X_SPEC)
SPLIT_H = _k4.components_ / _k4.components_.max(axis=1, keepdims=True)
SPLIT_MATCH = np.argmax(np.corrcoef(SPLIT_H, S_TRUE)[:4, 4:], axis=1)    # nearest true species

# (c) baseline offset added to every spectrum, k = 3
BASE_OFFSET = 0.3
_bm = NMF(n_components=3, init="nndsvda", max_iter=20000, tol=1e-9, random_state=0).fit(X_SPEC + BASE_OFFSET)
_bperm, BASE_CORR = match_components(_bm.components_, S_TRUE)
BASE_H = (_bm.components_ / _bm.components_.max(axis=1, keepdims=True))[_bperm]
BASE_FLOOR = float(BASE_H.min(axis=1).max())

# ------------------------------------- identifiability (graduate scene)
# rows of W normalised to sum 1 are points in a triangle (k = 3 simplex)
_srng = np.random.default_rng(12)
SIMPLEX_SPREAD = _srng.dirichlet([0.45] * 3, 160)     # touches the edges
SIMPLEX_CLUSTER = _srng.dirichlet([30.0] * 3, 160)     # bunched in the middle
SIMPLEX_V = np.array([[0.0, 1.0], [-np.sqrt(3) / 2, -0.5], [np.sqrt(3) / 2, -0.5]])


def _bary(P, V):
    T_ = np.column_stack([V[1] - V[0], V[2] - V[0]])
    l12 = np.linalg.solve(T_, (P - V[0]).T).T
    return np.column_stack([1 - l12.sum(axis=1), l12])


def _inner_triangle(P, angle=0.35):
    """Smallest rotated equilateral triangle (about the centroid) containing P."""
    c = P.mean(axis=0)
    u = np.array([[np.cos(angle + a), np.sin(angle + a)] for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
                                                                     np.pi / 2 + 4 * np.pi / 3)])
    lo, hi = 0.0, 2.0
    for _ in range(60):
        r = (lo + hi) / 2
        lo, hi = (r, hi) if (_bary(P, c + r * u) < 0).any() else (lo, r)
    return c + 1.05 * hi * u


SIMPLEX_CLUSTER_XY = SIMPLEX_CLUSTER @ SIMPLEX_V
SIMPLEX_SPREAD_XY = SIMPLEX_SPREAD @ SIMPLEX_V
SIMPLEX_INNER = _inner_triangle(SIMPLEX_CLUSTER_XY)
assert (_bary(SIMPLEX_INNER, SIMPLEX_V) >= 0).all(), "alternative triangle must stay inside"
assert (_bary(SIMPLEX_CLUSTER_XY, SIMPLEX_INNER) >= -1e-9).all()

# ------------------------------------------------ geometry (2 features)
# samples are non-negative mixtures of two parts; NMF finds the cone edges,
# PCA finds orthogonal variance directions through the mean.
_grng = np.random.default_rng(2)
GEO_PARTS = np.array([[1.0, 0.2], [0.25, 1.0]])
GEO_X = np.clip(_grng.gamma(2.0, 0.45, (40, 2)) @ GEO_PARTS + _grng.normal(0, 0.03, (40, 2)), 0, None)
_geo = NMF(n_components=2, init="nndsvda", max_iter=20000, tol=1e-9, random_state=0).fit(GEO_X)
GEO_H = _geo.components_ / np.linalg.norm(_geo.components_, axis=1, keepdims=True)
GEO_MEAN = GEO_X.mean(axis=0)
GEO_PC = np.linalg.svd(GEO_X - GEO_MEAN)[2]          # rows = PC1, PC2

# ---------------------------------------------------------------- drawing
class TextMatrix(VGroup):
    """A grid of numbers with square brackets; rows/columns are addressable."""

    def __init__(self, data, fmt="{:g}", font_size=34, h_buff=0.85, v_buff=0.6, **kwargs):
        super().__init__(**kwargs)
        self.n_rows, self.n_cols = len(data), len(data[0])
        self.cells = VGroup(*[
            T(fmt.format(x + 0.0), size=font_size) for row in data for x in row
        ])
        for idx, cell in enumerate(self.cells):
            i, j = divmod(idx, self.n_cols)
            cell.move_to(np.array([j * h_buff, -i * v_buff, 0]))
        self.cells.center()
        self.brackets = self._brackets(self.cells)
        self.add(self.cells, self.brackets)

    @staticmethod
    def _brackets(grid, pad=0.15, arm=0.12):
        top, bot = grid.get_top()[1] + pad, grid.get_bottom()[1] - pad
        left, right = grid.get_left()[0] - pad, grid.get_right()[0] + pad
        lb = VMobject().set_points_as_corners(
            [[left + arm, top, 0], [left, top, 0], [left, bot, 0], [left + arm, bot, 0]])
        rb = VMobject().set_points_as_corners(
            [[right - arm, top, 0], [right, top, 0], [right, bot, 0], [right - arm, bot, 0]])
        return VGroup(lb, rb).set_stroke(WHITE, 3)

    def entry(self, i, j):
        return self.cells[i * self.n_cols + j]

    def row(self, i):
        return VGroup(*[self.entry(i, j) for j in range(self.n_cols)])

    def col(self, j):
        return VGroup(*[self.entry(i, j) for i in range(self.n_rows)])


def spectrum_axes(y_max=1.1, x_length=9.0, y_length=4.2, y_label="absorbance"):
    """Wavelength axes with Text tick labels (no LaTeX)."""
    ax = Axes(x_range=[400, 700, 50], y_range=[0, y_max, y_max / 4],
              x_length=x_length, y_length=y_length, tips=False,
              axis_config={"include_numbers": False, "stroke_width": 2})
    ticks = VGroup(*[T(str(x), size=20).next_to(ax.c2p(x, 0), DOWN, buff=0.15)
                     for x in (400, 500, 600, 700)])
    xl = T("wavelength (nm)", size=22).next_to(ticks, DOWN, buff=0.15)
    yl = T(y_label, size=22).rotate(np.pi / 2).next_to(ax.y_axis, LEFT, buff=0.2)
    return VGroup(ax, ticks, xl, yl), ax


def curve(ax, x, y, color, width=3, opacity=1.0):
    g = ax.plot_line_graph(x, y, add_vertex_dots=False, line_color=color,
                           stroke_width=width)["line_graph"]
    return g.set_stroke(opacity=opacity)


def section_title(text):
    return T(text, size=40).to_edge(UP, buff=0.4)


LEVELS = [("beginner", "#82E0AA"), ("intermediate", "#F7DC6F"), ("advanced", "#F0B27A"),
          ("graduate", "#EC7063")]
WPS = 2.6            # narration speed: words per second (156 wpm)
CAP_SIZE = 24
CAP_CHARS = 76
STAGE_BOTTOM = -2.55  # keep scene content above this line (captions below)


def caption_chunks(text, width=CAP_CHARS):
    """Split narration into captions of at most two lines, breaking at
    sentence ends first, then at clause commas, then anywhere."""
    units = []
    for sent in re.split(r"(?<=[.?!;:])\s+", text):
        if len(textwrap.wrap(sent, width)) <= 2:
            units.append(sent)
            continue
        acc = ""
        for piece in re.split(r"(?<=,)\s+", sent):
            cand = f"{acc} {piece}".strip()
            if acc and len(textwrap.wrap(cand, width)) > 2:
                units.append(acc)
                acc = piece
            else:
                acc = cand
        units.append(acc)
    chunks, acc = [], ""
    for u in units:
        cand = f"{acc} {u}".strip()
        if acc and len(textwrap.wrap(cand, width)) > 2:
            chunks.append(acc)
            acc = u
        else:
            acc = cand
    chunks.append(acc)
    out = []
    for c in chunks:
        lines = textwrap.wrap(c, width)
        if len(lines) == 2:                       # balance the two lines
            for w in range(len(c) // 2, width + 1):
                if len(textwrap.wrap(c, w)) == 2:
                    lines = textwrap.wrap(c, w)
                    break
        out += ["\n".join(lines[i:i + 2]) for i in range(0, len(lines), 2)]
    return out


class NMFScene(Scene):
    """Scene paced by its narration.

    `say(text, *anims)` shows `text` as a burned-in caption (two lines at a
    time), plays `anims` while it is on screen and holds it for as long as it
    takes to read aloud at WPS words per second. Every caption is logged with
    its start/end time and speaker and written to media/captions/<Scene>.json,
    from which render_all.sh builds the .srt file and SCRIPT.md.
    """
    part = ""
    speaker = "A"
    level = 0                 # 1 beginner … 4 graduate (badge in the top-right corner)

    def setup(self):
        self.tag = VGroup()
        if self.part:
            self.tag.add(T(self.part, 16, color=GREY_B).to_corner(UL, buff=0.15))
        if self.level:
            name, col = LEVELS[self.level - 1]
            dots = "●" * self.level + "○" * (4 - self.level)
            self.tag.add(T(f"{dots}  {name}", 16, color=col).to_corner(UR, buff=0.15))
        self.add(self.tag)
        self.cap = None
        self.ref = None
        self.log = []

    def cite(self, text):
        """Small reference line, bottom-right above the captions."""
        ref = T(text, 16, color=GREY_B)
        ref.move_to([config.frame_width / 2 - 0.25 - ref.width / 2, STAGE_BOTTOM - 0.1, 0])
        anims = [FadeIn(ref)]
        if self.ref is not None and self.ref in self.mobjects:
            anims.append(FadeOut(self.ref))
        self.ref = ref
        return anims

    def _show_caption(self, text):
        if self.cap is not None:
            self.remove(self.cap)
        cap = T(text, CAP_SIZE, line_spacing=0.8)
        cap.add_background_rectangle(color=BLACK, opacity=0.8, buff=0.12)
        cap.to_edge(DOWN, buff=0.18)
        self.add(cap)
        self.cap = cap

    def say(self, text, *anims, run_time=None, who=None, hold=0.0):
        for i, chunk in enumerate(caption_chunks(text)):
            dur = len(chunk.split()) / WPS + 0.25
            t0 = self.renderer.time
            self._show_caption(chunk)
            if i == 0 and anims:
                self.play(*anims, run_time=run_time or min(1.5, dur))
            rest = dur - (self.renderer.time - t0)
            if rest > 0.04:
                self.wait(rest)
            self.log.append({"start": t0, "end": self.renderer.time, "who": who or self.speaker,
                             "text": chunk.replace("\n", " ")})
        if hold:
            self.wait(hold)

    def clear_stage(self, *keep, **kw):
        """Fade out everything except the part tag, the caption and `keep`."""
        keep_ids = {id(m) for m in keep} | {id(self.tag), id(self.cap)}
        out = [FadeOut(m) for m in self.mobjects if id(m) not in keep_ids]
        if out:
            self.play(*out, **kw)

    def finish(self):
        self.wait(0.6)
        os.makedirs("media/captions", exist_ok=True)
        with open(f"media/captions/{type(self).__name__}.json", "w") as f:
            json.dump({"duration": self.renderer.time, "captions": self.log}, f, indent=1)
