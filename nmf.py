"""NMF explainer: 13 narrated Manim scenes in four parts (~14.5 min).

  Part 1  How the transform works      S01–S05   (speaker A; S05 B)
  Part 2  Biochemical use case         S06–S08   (speaker B)
  Part 3  Primary literature example   S09–S11   (A, then B)
  Part 4  Strengths & limitations      S12–S14   (A and B)

A badge in the top-right corner marks the level of each scene, rising from
beginner (S01) to graduate (S13–S14); on-screen citations point to the papers.

Each scene is paced by its narration: `self.say(text, *anims)` burns the text
in as a caption, plays the animations and holds for as long as the text takes
to read aloud. render_all.sh turns the logged captions into an .srt file and
SCRIPT.md, so the narration lives in exactly one place: this file.

Render one scene:   ./render.sh S06Kinetics -ql
Render everything:  ./render_all.sh

All on-screen numbers come from helpers.py (numpy / scikit-learn); the only
hand-typed numbers are the published counts from Alexandrov et al. (2013).
Needs LaTeX (latex + dvisvgm) for the maths.
"""
import numpy as np
from manim import *

from helpers import *
from pictures import (cigarette, clock, cuvette, dna, enzyme, mini_kinetics, mini_spectrum, neuron,
                      spectrometer, stopped_flow, sun, tissue, tumour_cell)

SCENES = ["S01Intro", "S02Input", "S03Output", "S04Fitting", "S05KeptLost",
          "S06Kinetics", "S07Knobs", "S08Questions",
          "S09Paper", "S10Method", "S11KeyFigure",
          "S12StrengthsLimits", "S13Theory", "S14Conclusions"]
SVD1, SVD2, NEG = PURPLE_B, TEAL_B, RED
P1 = "Part 1 · How the transform works"
P2 = "Part 2 · Biochemical use case"
P3 = "Part 3 · Primary literature"
P4 = "Part 4 · Strengths & limitations"
TOP = 2.95                    # top of the content area (below a section title)


def fit_width(mob, width=config.frame_width - 1.0):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob


def bullet_list(lines, size=30, buff=0.35, colors=None, mark="•"):
    g = VGroup(*[T(f"{mark}  {s}", size, t2c=colors or {}) for s in lines])
    return g.arrange(DOWN, aligned_edge=LEFT, buff=buff)


def bars(values, colors, unit=0.5, width=0.5, gap=0.3):
    """Vertical bars from a common baseline (negative values point down)."""
    if not isinstance(colors, (list, tuple)):
        colors = [colors] * len(values)
    g = VGroup()
    for j, (v, c) in enumerate(zip(values, colors)):
        h = max(abs(v) * unit, 0.02)
        r = Rectangle(width=width, height=h, stroke_width=0, fill_color=c, fill_opacity=0.9)
        r.move_to([j * (width + gap), np.sign(v or 1) * h / 2, 0])
        g.add(r)
    base = Line(g.get_left() + LEFT * 0.15, g.get_right() + RIGHT * 0.15, stroke_width=2)
    base.set_y(0)
    return VGroup(base, g)


def grid(mat, color, cell=0.42, vmax=None):
    """Heat-map of a small matrix in one colour (opacity = value)."""
    vmax = vmax or mat.max()
    g = VGroup(*[Square(side_length=cell, stroke_color=GREY_D, stroke_width=1, fill_color=color,
                        fill_opacity=float(np.clip(mat[i, j] / vmax, 0, 1)))
                 .move_to([j * cell, -i * cell, 0]) for i in range(mat.shape[0]) for j in range(mat.shape[1])])
    return g.center()


def sig_bars(p, width=6.0, height=0.8, header=True):
    """96-channel mutational profile, coloured by substitution type."""
    bw = width / 96
    top = p.max()
    g = VGroup()
    for i, v in enumerate(p):
        h = max(v / top * height, 0.004)
        g.add(Rectangle(width=bw * 0.75, height=h, stroke_width=0, fill_opacity=1,
                        fill_color=SUB_COLORS[i // 16]).move_to([-width / 2 + (i + 0.5) * bw, h / 2, 0]))
    out = VGroup(Line([-width / 2, 0, 0], [width / 2, 0, 0], stroke_width=1, color=GREY_B), g)
    if header:
        strip = VGroup(*[Rectangle(width=width / 6 * 0.96, height=0.07, stroke_width=0, fill_opacity=1,
                                   fill_color=SUB_COLORS[j]).move_to([-width / 2 + (j + 0.5) * width / 6,
                                                                      height + 0.1, 0])
                         for j in range(6)])
        labs = VGroup(*[T(SUBS[j], 14).next_to(strip[j], UP, buff=0.05) for j in range(6)])
        out.add(strip, labs)
    return out


def small_axes(x_range, y_range, x_length, y_length, xticks, yticks, xlab, ylab, tick_size=16):
    ax = Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, tips=False,
              axis_config={"include_numbers": False, "stroke_width": 2})
    xt = VGroup(*[T(f"{x:g}", tick_size).next_to(ax.c2p(x, y_range[0]), DOWN, buff=0.1) for x in xticks])
    yt = VGroup(*[T(f"{y:g}", tick_size).next_to(ax.c2p(x_range[0], y), LEFT, buff=0.1) for y in yticks])
    xl = T(xlab, 18).next_to(xt, DOWN, buff=0.1)
    yl = T(ylab, 18).next_to(ax, UP, buff=0.12)
    return VGroup(ax, xt, yt, xl, yl), ax


def VWH(size=60):
    eq = MathTex(r"V", r"\approx", r"W", r"H", font_size=size)
    eq[2].set_color(C1)
    eq[3].set_color(C2)
    return eq


# ================================================================ PART 1
class S01Intro(NMFScene):
    level = 1
    def construct(self):
        spec = spectrometer().scale(0.85).move_to(UP * 2.0)
        grp, ax = spectrum_axes(y_max=3.0, x_length=7.0, y_length=2.6)
        grp.move_to(DOWN * 1.0)
        mixes = VGroup(*[curve(ax, WL, X_SPEC[i], GREY_B, 3) for i in (2, 9, 17)])
        self.say("A spectrometer shines light through a sample and records how much light is absorbed "
                 "at every wavelength.", FadeIn(spec, lag_ratio=0.1), run_time=2)
        self.say("Our samples are mixtures, so each spectrum is a sum of overlapping bands, and we cannot "
                 "simply read off what is inside.", Create(grp[0]), FadeIn(grp[1:]),
                 LaggedStart(*[Create(m) for m in mixes], lag_ratio=0.3), run_time=3)

        self.clear_stage(run_time=0.6)
        name = T("Non-negative Matrix Factorization (NMF)", 46).to_edge(UP, buff=0.6)
        eq = VWH(64).next_to(name, DOWN, buff=0.4)
        self.say("Non-negative matrix factorization, or NMF, pulls such mixtures apart into pure parts, "
                 "and the amount of each part in every sample.", Write(name), FadeIn(eq, shift=UP * 0.2), run_time=2)
        self.say("It began in chemistry, as positive matrix factorization by Paatero and Tapper in 1994, and was "
                 "popularised by Lee and Seung in Nature in 1999.",
                 *self.cite("Paatero & Tapper 1994, Environmetrics · Lee & Seung 1999, Nature"))
        qs = VGroup(T("1   What goes in?", 30), T("2   What comes out?", 30),
                    T("3   What is kept, and what is lost?", 30),
                    T("4   Which biochemical question becomes easier?", 30)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).next_to(eq, DOWN, buff=0.55)
        self.say("By the end you should be able to answer four questions. What goes in? What comes out? "
                 "What is kept, and what is lost? And which biochemical question becomes easier?",
                 LaggedStart(*[FadeIn(q, shift=RIGHT * 0.2) for q in qs], lag_ratio=0.6), run_time=5)
        self.finish()


class S02Input(NMFScene):
    level = 1
    part = P1

    def construct(self):
        title = section_title("What goes in: a non-negative matrix V")
        V = TextMatrix(V_DATA, font_size=34, h_buff=1.45, v_buff=0.65)
        n, m = V_DATA.shape
        rl = VGroup(*[T(f"sample {i + 1}", 22).next_to(V.row(i), LEFT, buff=0.5) for i in range(n)])
        cl = VGroup(*[T(f"feature {j + 1}", 18).next_to(V.col(j), UP, buff=0.35) for j in range(m)])
        for j, lab in enumerate(cl):
            lab.set_x(V.entry(0, j).get_x())
        toy = VGroup(V, rl, cl).move_to([-3.4, 0.6, 0])
        m1 = M(r"V \in \mathbb{R}_{\ge 0}^{\,n \times m}", 48).move_to([3.4, 1.4, 0])
        m2 = M(r"V_{ij} = \text{feature } j \text{ of sample } i", 32).next_to(m1, DOWN, buff=0.45)
        m3 = M(r"V_{ij} \ge 0 \quad \text{for all } i, j", 36, color=GREEN).next_to(m2, DOWN, buff=0.4)
        self.say("What goes in is a matrix V with n rows and m columns. Each row is a sample, and each "
                 "column is a feature, such as one wavelength.", Write(title), FadeIn(toy), Write(m1), run_time=2)
        rbox = SurroundingRectangle(VGroup(rl[0], V.row(0)), color=YELLOW, buff=0.08)
        cbox = SurroundingRectangle(VGroup(cl[1], V.col(1)), color=TEAL, buff=0.08)
        self.say("The only requirement is that every entry is zero or positive. Absorbances, counts and "
                 "intensities all qualify.", Create(rbox), Create(cbox), FadeIn(m2), FadeIn(m3))

        self.clear_stage(title, run_time=0.6)
        cuv = cuvette().scale(1.3).move_to([-4.6, 0.4, 0])
        beam = Line(cuv.get_left() + LEFT * 1.2, cuv.get_right() + RIGHT * 0.8, color=YELLOW, stroke_width=5)
        lab = T("path length ℓ", 20, color=GREY_A).next_to(cuv, DOWN, buff=0.2)
        bl1 = M(r"A_{ij} \;=\; \ell \sum_{s=1}^{k} c_{is}\,\varepsilon_{sj}", 46).move_to([2.2, 1.4, 0])
        bl2 = M(r"A \;=\; \ell\; C\, E", 52).move_to([2.2, -0.3, 0])
        cl_ = M(r"C \in \mathbb{R}_{\ge0}^{\,n\times k}\ \text{concentrations}", 28, color=C1)
        el_ = M(r"E \in \mathbb{R}_{\ge0}^{\,k\times m}\ \text{molar absorptivities}", 28, color=C2)
        VGroup(cl_, el_).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(bl2, DOWN, buff=0.4)
        self.say("For spectra this structure comes straight from physics. By the Beer–Lambert law, the "
                 "absorbance of sample i at wavelength j is the path length times a sum over species of "
                 "concentration times molar absorptivity.",
                 FadeIn(cuv), Create(beam), FadeIn(lab), Write(bl1), run_time=2.5)
        self.say("In matrix form, A equals C times E. Both factors are non-negative, so a stack of mixture "
                 "spectra is exactly a non-negative matrix product.",
                 TransformFromCopy(bl1, bl2), FadeIn(cl_), FadeIn(el_), run_time=2)

        self.clear_stage(title, run_time=0.6)
        sub = X_SPEC[:, ::5]
        cw = 0.12
        vmax = sub.max()
        heat = VGroup(*[
            Square(side_length=cw, stroke_width=0, fill_opacity=1,
                   fill_color=interpolate_color(BLACK, WHITE, (sub[i, j] / vmax) ** 0.7))
            .move_to([j * cw, -i * cw, 0])
            for i in range(sub.shape[0]) for j in range(sub.shape[1])
        ]).center().move_to([0, 0.5, 0])
        frame = SurroundingRectangle(heat, color=GREY_B, buff=0.03, stroke_width=2)
        n2, m2_ = X_SPEC.shape
        hl = M(rf"V \in \mathbb{{R}}_{{\ge0}}^{{\,{n2}\times {m2_}}}", 36).next_to(frame, DOWN, buff=0.25)
        self.say(f"Our example data: {n2} mixture spectra at {m2_} wavelengths, {X_SPEC.size:,} non-negative "
                 f"numbers.",
                 FadeIn(heat, lag_ratio=0.002), Create(frame), Write(hl), run_time=2.5)
        self.finish()


class S03Output(NMFScene):
    level = 1
    part = P1

    def construct(self):
        title = section_title("What comes out:  V ≈ W H")
        V = TextMatrix(V_DATA)
        W = TextMatrix(W_DATA, fmt="{:.1f}", h_buff=1.15)
        H = TextMatrix(H_DATA, fmt="{:.1f}", h_buff=1.15)
        eq = VGroup(V, T("≈", 44), W, T("×", 44), H).arrange(RIGHT, buff=0.4)
        eq.scale_to_fit_width(10.5).move_to([0, 1.3, 0])
        dims = M(r"W \in \mathbb{R}_{\ge0}^{\,n\times k}, \qquad H \in \mathbb{R}_{\ge0}^{\,k\times m}, "
                 r"\qquad k < \min(n, m)", 38).move_to([0, -0.8, 0])
        self.say("What comes out are two smaller non-negative matrices: W, with n rows and k columns, and H, "
                 "with k rows and m columns, chosen so that V is approximately W times H.",
                 Write(title), FadeIn(eq), Write(dims), run_time=2.5)
        labs = VGroup(T("W = amounts", 24, color=C1).next_to(W, DOWN, buff=0.2),
                      T("H = parts", 24, color=C2).next_to(H, DOWN, buff=0.2))
        self.say("Rows of H are the parts, for example pure spectra. Rows of W are the amounts: how much of "
                 "each part every sample contains.",
                 W.col(0).animate.set_color(C1), H.row(0).animate.set_color(C1),
                 W.col(1).animate.set_color(C2), H.row(1).animate.set_color(C2), FadeIn(labs))

        w0, h0 = W_DATA[0], H_DATA[:, 0]
        entry = M(r"(WH)_{ij} = \sum_{r=1}^{k} W_{ir}\,H_{rj}", 40).move_to([0, -0.9, 0])
        calc = M(rf"(WH)_{{11}} = {w0[0]:.1f}\cdot{h0[0]:.1f} + {w0[1]:.1f}\cdot{h0[1]:.1f} = "
                 rf"{float(w0 @ h0):.2f} \approx {V_DATA[0, 0]:g}", 38, color=YELLOW).next_to(entry, DOWN, buff=0.35)
        boxes = VGroup(SurroundingRectangle(W.row(0), color=YELLOW, buff=0.08),
                       SurroundingRectangle(H.col(0), color=YELLOW, buff=0.08),
                       SurroundingRectangle(V.entry(0, 0), color=YELLOW, buff=0.08))
        self.say(f"Each entry of the product is a dot product of a row of W with a column of H. Here "
                 f"{w0[0]:.1f} times {h0[0]:.1f} plus {w0[1]:.1f} times {h0[1]:.1f} gives "
                 f"{float(w0 @ h0):.2f}, close to the measured {V_DATA[0, 0]:g}.",
                 FadeOut(labs), ReplacementTransform(dims, entry), Create(boxes), Write(calc), run_time=2.5)

        # sum of rank-one layers
        self.clear_stage(title, run_time=0.6)
        L1 = np.outer(W_DATA[:, 0], H_DATA[0])
        L2 = np.outer(W_DATA[:, 1], H_DATA[1])
        vmax = max(L1.max(), L2.max(), WH_DATA.max())
        g1, g2, g3 = grid(L1, C1, vmax=vmax), grid(L2, C2, vmax=vmax), grid(WH_DATA, WHITE, vmax=vmax)
        row = VGroup(g1, M(r"+", 50), g2, M(r"=", 50), g3).arrange(RIGHT, buff=0.5).move_to([0, 0.2, 0])
        under = VGroup(M(r"\mathbf w_1 \mathbf h_1^{\top}", 34, color=C1).next_to(g1, DOWN, buff=0.25),
                       M(r"\mathbf w_2 \mathbf h_2^{\top}", 34, color=C2).next_to(g2, DOWN, buff=0.25),
                       M(r"WH", 34).next_to(g3, DOWN, buff=0.25))
        outer = M(r"WH \;=\; \sum_{r=1}^{k} \mathbf w_r\, \mathbf h_r^{\top}"
                  r"\qquad (\mathbf w_r = \text{column } r \text{ of } W,\ \mathbf h_r^{\top} = \text{row } r \text{ of } H)",
                  32).move_to([0, 2.4, 0])
        fit_width(outer)
        self.say("Equivalently, W times H is a sum of k rank-one layers, each an amount column times a part "
                 "row. Because nothing is negative, the layers can only add up; they never cancel.",
                 Write(outer), FadeIn(row, lag_ratio=0.2), FadeIn(under), run_time=3)
        full, small = SPEC_COMPRESS
        n, m = X_SPEC.shape
        rank = M(rf"\operatorname{{rank}}(WH) \le k, \qquad \underbrace{{k(n+m)}}_{{{small}}} \;\ll\; "
                 rf"\underbrace{{nm}}_{{{full}}}", 40).move_to([0, -1.9, 0])
        self.say(f"So W times H has rank at most k, and it is stored in k times n plus m numbers instead of n "
                 f"times m: for our {n} spectra with k = 3, {small:,} instead of {full:,}, about "
                 f"{full / small:.0f} times fewer.", Write(rank), run_time=2)
        self.finish()


class S04Fitting(NMFScene):
    level = 2
    part = P1

    def construct(self):
        title = section_title("How W and H are found")
        e1 = M(r"\min_{W \ge 0,\; H \ge 0}\; \tfrac12 \lVert V - WH \rVert_F^2 \;=\; "
               r"\tfrac12 \sum_{i,j}\big(V_{ij}-(WH)_{ij}\big)^2", 40).move_to([0, 1.7, 0])
        self.say("W and H are found by optimization: minimize half the squared Frobenius distance between V "
                 "and W times H, with both factors constrained to be non-negative.",
                 Write(title), Write(e1), run_time=2.5)
        e2 = M(r"D(V\,\|\,WH) = \sum_{i,j}\Big(V_{ij}\log\frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij}\Big)",
               38).move_to([0, 0.0, 0])
        e3 = M(r"-\log p(V \mid WH) = \tfrac{1}{2\sigma^2}\lVert V-WH\rVert_F^2 + c \;\;(\text{Gaussian}),"
               r"\qquad D(V\|WH) + c \;\;(\text{Poisson})", 30, color=GREY_A).move_to([0, -1.6, 0])
        fit_width(e3)
        self.say("Squared error is the negative log-likelihood under Gaussian noise. For count data we use "
                 "the generalized Kullback–Leibler divergence instead, which is the negative Poisson "
                 "log-likelihood up to a constant.", FadeIn(e2), FadeIn(e3), run_time=2)

        self.clear_stage(title, run_time=0.6)
        g1 = M(r"\nabla_H \;=\; W^{\top}WH - W^{\top}V", 38)
        g2 = M(r"H \leftarrow H - \eta \odot \big(W^{\top}WH - W^{\top}V\big),\quad \eta = \frac{H}{W^{\top}WH}", 36)
        g3 = M(r"H \leftarrow H \odot \frac{W^{\top}V}{W^{\top}WH}, \qquad W \leftarrow W \odot \frac{VH^{\top}}{WHH^{\top}}",
               38, color=GREEN)
        col = VGroup(g1, g2, g3).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        fit_width(col, 7.6).to_edge(LEFT, buff=0.5).set_y(1.0)
        box3 = SurroundingRectangle(g3, color=GREEN, buff=0.12)
        self.say("The gradient with respect to H is W-transpose W H minus W-transpose V.", Write(g1))
        self.say("Take a gradient step with its own step size for every entry, and choose eta equal to H "
                 "divided element-wise by W-transpose W H.", Write(g2), run_time=2)
        self.say("The H terms cancel and the subtraction becomes a ratio. This is Lee and Seung's "
                 "multiplicative update, and the same algebra gives the update for W.",
                 Write(g3), Create(box3), *self.cite("Lee & Seung 2001, NIPS 13"), run_time=2)

        n_show = 50
        grp, ax = small_axes([0, n_show, 10], [0, 30, 10], 4.4, 3.0, range(0, n_show + 1, 10),
                             (0, 10, 20, 30), "iteration", "½‖V − WH‖²  (toy V, random start)")
        grp.move_to([4.2, 0.9, 0])
        line = ax.plot_line_graph(np.arange(n_show + 1), MU_LOSS[:n_show + 1], add_vertex_dots=False,
                                  line_color=YELLOW, stroke_width=3)["line_graph"]
        self.say(f"Every factor is non-negative, so W and H stay non-negative without clipping, and the loss "
                 f"provably never increases. On our toy matrix it falls from {MU_LOSS[0]:.1f} to "
                 f"{MU_LOSS[MU_N]:.2f}.", FadeIn(grp), Create(line, rate_func=linear), run_time=3)
        kkt = M(r"H \ge 0,\quad \nabla_H \ge 0,\quad H \odot \nabla_H = 0", 38, color=YELLOW).move_to([0, -1.5, 0])
        self.say("At a fixed point the Karush–Kuhn–Tucker conditions hold: wherever H is positive, its gradient "
                 "is zero. But the loss is not jointly convex, so different starts reach different local minima.",
                 Write(kkt), *self.cite("convergence: Lin 2007, Neural Comput. & IEEE TNN"), run_time=2)
        self.finish()


class S05KeptLost(NMFScene):
    level = 2
    part = P1
    speaker = "B"

    def construct(self):
        title = section_title("Geometric view: what is kept, what is lost")
        lim = float(np.ceil(GEO_X.max() * 1.05 * 2) / 2)     # axis range covers every sample
        ax = Axes(x_range=[0, lim, 1], y_range=[0, lim, 1], x_length=4.6, y_length=4.6, tips=False,
                  axis_config={"include_numbers": False, "stroke_width": 2}).move_to([-4.0, 0.2, 0])
        axl = VGroup(T("feature 1", 18).next_to(ax.x_axis, DOWN, buff=0.15),
                     T("feature 2", 18).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.15))
        dots = VGroup(*[Dot(ax.c2p(*p), radius=0.05, color=GREY_B) for p in GEO_X])
        ends = [GEO_H[r] * lim / GEO_H[r].max() for r in range(2)]
        rays = VGroup(*[Arrow(ax.c2p(0, 0), ax.c2p(*e), buff=0, color=COMP_COLORS[r], stroke_width=5,
                              max_tip_length_to_length_ratio=0.06) for r, e in enumerate(ends)])
        cone = Polygon(ax.c2p(0, 0), ax.c2p(*ends[0]), ax.c2p(lim, lim), ax.c2p(*ends[1]),
                       stroke_width=0, fill_color=YELLOW, fill_opacity=0.10)
        hl = VGroup(M(r"\mathbf h_1", 30, color=C1).next_to(rays[0].get_end(), UP, buff=0.1),
                    M(r"\mathbf h_2", 30, color=C2).next_to(rays[1].get_end(), RIGHT, buff=0.1))
        self.say("Geometrically, each sample is a point in feature space, here with just two features. NMF "
                 "finds k non-negative directions, the parts, whose cone contains the data.",
                 Write(title), Create(ax), FadeIn(axl), FadeIn(dots, lag_ratio=0.02), run_time=2)
        self.play(FadeIn(cone), GrowArrow(rays[0]), GrowArrow(rays[1]), FadeIn(hl), run_time=1.2)
        x0 = 0.4
        e1 = M(r"\mathbf v_i \approx W_{i1}\,\mathbf h_1 + W_{i2}\,\mathbf h_2, \qquad W_{ir} \ge 0", 36)
        e1.move_to([x0 + e1.width / 2, 2.5, 0])
        self.say("Every point is a non-negative combination of the parts, which is exactly the Beer–Lambert "
                 "picture of a mixture.", Write(e1))
        pcs = VGroup(*[DashedLine(ax.c2p(*(GEO_MEAN - 1.6 * GEO_PC[r])), ax.c2p(*(GEO_MEAN + 1.6 * GEO_PC[r])),
                                  color=SVD1, stroke_width=3) for r in range(2)])
        e2 = M(r"V - \bar V \approx U_k\Sigma_k Q_k^{\top}, \qquad Q_k^{\top}Q_k = I", 36, color=SVD1)
        e2.next_to(e1, DOWN, buff=0.3).align_to(e1, LEFT)
        self.say("PCA instead centres the data and finds orthogonal directions of maximal variance. They "
                 "leave the positive quadrant, so samples need negative coefficients.",
                 Create(pcs), Write(e2), run_time=2)

        kept = VGroup(T("Kept / emphasised", 26, color=GREEN),
                      *bullet_list(["additive, parts-based structure", "the main rank-k patterns"],
                                   size=22, buff=0.12)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        lost = VGroup(T("Lost / distorted", 26, color=RED_B),
                      *bullet_list(["residual: noise + anything beyond k parts",
                                    "absolute scale and order of the parts",
                                    "small or rare components"], size=22, buff=0.12)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        kl = VGroup(kept, lost).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(e2, DOWN, buff=0.3)
        kl.align_to(e1, LEFT)
        self.say("So NMF keeps additive, parts-based structure and the main rank-k patterns. It loses the "
                 "residual, which is noise plus anything beyond k parts, the absolute scale and order of the "
                 "parts, and possibly small, rare components.", FadeIn(kl, lag_ratio=0.2), run_time=2.5)
        ey = M(rf"\min_{{\operatorname{{rank}} X \le k}}\lVert V-X\rVert_F^2 = {SVD_FRO2:.2f} \;\le\; "
               rf"\lVert V-WH\rVert_F^2 = {NMF_FULL_FRO2:.2f}", 32, color=YELLOW)
        fit_width(ey, config.frame_width / 2 - x0 - 0.2).next_to(kl, DOWN, buff=0.25).align_to(e1, LEFT)
        if ey.get_bottom()[1] < STAGE_BOTTOM:
            VGroup(e1, e2, kl, ey).shift(UP * (STAGE_BOTTOM - ey.get_bottom()[1]))
        self.say(f"It also gives up a little fit. By the Eckart–Young theorem, truncated SVD is the best "
                 f"rank-k approximation: {SVD_FRO2:.2f} versus {NMF_FULL_FRO2:.2f} for NMF on our toy matrix.",
                 Write(ey))
        self.finish()


# ================================================================ PART 2
class S06Kinetics(NMFScene):
    level = 2
    part = P2
    speaker = "B"

    def construct(self):
        title = section_title("Use case: catching a reaction intermediate")
        sf = stopped_flow().scale(0.95).move_to([-3.6, 0.2, 0])
        self.say("Why not just look at the raw data? In a stopped-flow experiment two solutions are mixed "
                 "within milliseconds, and absorption spectra are recorded while the reaction runs.",
                 Write(title), FadeIn(sf, lag_ratio=0.1), run_time=2.5)
        r1 = M(r"A \xrightarrow{\;k_1\;} B \xrightarrow{\;k_2\;} C", 46).move_to([3.4, 2.0, 0])
        r2 = M(r"[A](t) = e^{-k_1 t}", 34)
        r3 = M(r"[B](t) = \frac{k_1}{k_2-k_1}\big(e^{-k_1 t}-e^{-k_2 t}\big)", 34)
        r4 = M(r"[C](t) = 1 - [A](t) - [B](t)", 34)
        rr = VGroup(r2, r3, r4).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(r1, DOWN, buff=0.5)
        self.say("We simulate a two-step reaction: A turns into an intermediate B, which turns into the "
                 "product C.", Write(r1))
        self.say("With first-order steps, A decays as a single exponential, and B rises and falls as a "
                 "difference of two exponentials.", FadeIn(rr, lag_ratio=0.3), run_time=2)

        self.clear_stage(title, run_time=0.6)
        grp, ax = spectrum_axes(y_max=1.1, x_length=7.0, y_length=3.9)
        grp.move_to([-2.9, 0.1, 0])
        early, late = BLUE_A, PINK
        raw = VGroup(*[curve(ax, WL, KIN_X[i], interpolate_color(early, late, i / (KIN_T.size - 1)), 1.6, 0.8)
                       for i in range(KIN_T.size)])
        key = VGroup(T("early", 22, color=early), T("late", 22, color=late)).arrange(RIGHT, buff=0.4)
        key.next_to(ax, UP, buff=0.1).align_to(ax, RIGHT)
        self.say(f"Here are the data: {KIN_T.size} spectra over {KIN_T[-1]:.0f} seconds, coloured from early "
                 "to late. The bands overlap, and B is never present on its own, so its spectrum cannot be "
                 "measured directly.", Create(grp[0]), FadeIn(grp[1:]), FadeIn(key),
                 LaggedStart(*[Create(r) for r in raw], lag_ratio=0.06), run_time=4)
        qs = VGroup(T("Raw data cannot tell us:", 24, color=YELLOW),
                    T("• how many species?", 22), T("• is there an intermediate?", 22),
                    T("• what does it look like?", 22), T("• how fast is each step?", 22)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        qs.next_to(grp, RIGHT, buff=0.5).set_y(0.6)
        self.say("From the raw spectra we cannot tell how many species there are, whether an intermediate "
                 "exists, what it looks like, or how fast each step is.", FadeIn(qs, lag_ratio=0.3), run_time=2.5)

        self.clear_stage(title, run_time=0.6)
        bl = M(r"V_{t\lambda} = \sum_{s \in \{A,B,C\}} [s](t)\;\varepsilon_s(\lambda)"
               r"\quad\Longrightarrow\quad V = \underbrace{C(t)}_{W}\;\underbrace{E}_{H},\ \ k = 3", 36)
        fit_width(bl).move_to([0, 2.3, 0])
        self.say("By Beer–Lambert the data matrix is concentrations over time times pure spectra, so V equals "
                 "W times H with k equal to three. NMF needs no kinetic model to find them.", Write(bl),
                 *self.cite("chemometrics relative: MCR-ALS, de Juan & Tauler 2006"), run_time=2.5)

        sgrp, sax = spectrum_axes(y_max=1.1, x_length=5.8, y_length=3.0, y_label="normalised")
        sgrp.move_to([-3.5, -0.6, 0])
        kgrp, kax = small_axes([0, 100, 20], [0, 1.1, 0.5], 5.0, 3.0, (0, 20, 40, 60, 80, 100),
                               (0, 0.5, 1), "time (s)", "W (scaled)", tick_size=18)
        kgrp.move_to([3.6, -0.55, 0])
        truth = VGroup(*[DashedVMobject(curve(sax, WL, KIN_S[c], COMP_COLORS[c], 3), num_dashes=50)
                         for c in range(3)])
        est = VGroup(*[curve(sax, WL, KIN_S_EST[c], COMP_COLORS[c], 4) for c in range(3)])
        slabs = VGroup(*[T(n_, 22, color=COMP_COLORS[c]).next_to(sax.c2p(WL[np.argmax(KIN_S_EST[c])], 1.0), UP,
                                                                 buff=0.06) for c, n_ in enumerate("ABC")])
        hl = T("H: dashed = true, solid = NMF", 18, color=GREY_A).next_to(sgrp, UP, buff=0.12)
        rs = ", ".join(f"{r:.3f}" for r in KIN_SPEC_CORR)
        self.say(f"H holds the pure spectra. Solid lines are what NMF recovered and dashed lines are the truth "
                 f"used in the simulation, with correlations of {rs}, including the spectrum of B, which was "
                 f"never measured alone.", FadeIn(sgrp), FadeIn(hl), Create(truth), Create(est), FadeIn(slabs),
                 run_time=3)
        W_show = KIN_W_EST / KIN_AMP[None, :]
        dots = VGroup(*[Dot(kax.c2p(KIN_T[i], W_show[i, c]), radius=0.035, color=COMP_COLORS[c])
                        for c in range(3) for i in range(KIN_T.size)])
        self.say("W holds the time courses. A decays, B rises and falls, and C grows: the shape of a "
                 "consecutive reaction, found without telling NMF the mechanism.",
                 FadeIn(kgrp), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.005), run_time=2.5)
        tf = np.linspace(0, 100, 201)
        fitc = kin_conc(tf, KIN_K1_EST, KIN_K2_EST)
        fits = VGroup(*[curve(kax, tf, fitc[:, c], COMP_COLORS[c], 2.5) for c in range(3)])
        kf = M(rf"\hat k_1 = {KIN_K1_EST:.3f}\ (\text{{true }}{KIN_K1:g}),\qquad "
               rf"\hat k_2 = {KIN_K2_EST:.4f}\ (\text{{true }}{KIN_K2:g})\ \ \mathrm{{s}}^{{-1}}", 34, color=YELLOW)
        fit_width(kf).move_to(bl)
        self.say(f"Fitting the kinetic model to W, with a free amplitude for each species because W has no "
                 f"absolute scale, gives k1 = {KIN_K1_EST:.3f} per second against a true {KIN_K1:g}, and k2 = "
                 f"{KIN_K2_EST:.4f} against a true {KIN_K2:g}.",
                 Create(fits), ReplacementTransform(bl, kf), run_time=2.5)
        full, small = KIN_COMPRESS
        self.say(f"So the transform turns {full:,} overlapping numbers into {small:,} interpretable ones and a "
                 f"direct answer: one intermediate, absorbing at {KIN_B_PEAK:.0f} nanometres, forming at k1 and "
                 f"decaying at k2.", Indicate(est[1], color=C2), Indicate(slabs[1], color=C2))
        self.finish()


class S07Knobs(NMFScene):
    level = 3
    part = P2
    speaker = "B"

    def construct(self):
        title = section_title("The researcher's choices (the 'knobs')")
        pca = VGroup(T("PCA: few knobs", 28, color=SVD1),
                     *bullet_list(["k", "centre / scale?"], size=22, buff=0.15)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        pca.move_to([-5.4, 1.6, 0])
        self.say("That result depended on choices. PCA has few knobs: the number of components, and whether "
                 "to centre and scale the data.", Write(title), FadeIn(pca), run_time=1.5)
        fin = INIT_RANDOM[:, -1]
        rows = [
            ("k, the number of parts",
             T(f"error elbow + stability across restarts: {K_STAB[2]:.3f} at k = 3, {K_STAB[3]:.3f} at k = 4", 20, color=GREY_A)),
            ("loss (β-divergence)",
             M(r"\beta = 2:\ \text{squared error},\quad \beta = 1:\ \text{KL},\quad \beta = 0:\ \text{Itakura--Saito}", 28, color=GREY_A)),
            ("initialization + restarts",
             T(f"6 random starts end at {fin.min():.2f}–{fin.max():.2f};  NNDSVDa start: {INIT_NNDSVD[-1]:.2f}", 20, color=GREY_A)),
            ("regularization",
             M(r"+\;\alpha\Big(\rho\,(\lVert W\rVert_1+\lVert H\rVert_1) + \tfrac{1-\rho}{2}(\lVert W\rVert_F^2+\lVert H\rVert_F^2)\Big)",
               28, color=GREY_A)),
            ("preprocessing", T("baseline removal, normalization, which features to keep", 20, color=GREY_A)),
            ("solver + stopping rule", T("multiplicative updates vs coordinate descent; tolerance, max iterations", 20, color=GREY_A)),
        ]
        nmf = VGroup(T("NMF: more knobs", 28, color=C1))
        for head, ev in rows:
            r = VGroup(T(f"•  {head}", 24), ev).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            ev.shift(RIGHT * 0.4)
            nmf.add(r)
        nmf.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        fit_box(nmf, 9.4, TOP - STAGE_BOTTOM - 0.1)
        nmf.move_to([-2.3 + nmf.width / 2, (TOP + STAGE_BOTTOM) / 2, 0])
        self.say(f"NMF has more. First k, chosen from the elbow of the error curve and the stability of the "
                 f"parts across restarts: {K_STAB[2]:.3f} at k = 3 but {K_STAB[3]:.3f} at k = 4.",
                 FadeIn(nmf[0]), FadeIn(nmf[1]), *self.cite("consensus choice of k: Brunet et al. 2004, PNAS"))
        self.say("Second, the loss. The beta-divergence family includes squared error at beta equals two, KL "
                 "at one and Itakura–Saito at zero; it should match the noise.", FadeIn(nmf[2]),
                 *self.cite("Févotte & Idier 2011, Neural Comput."))
        self.say(f"Third, initialization: six random starts ended between {fin.min():.2f} and {fin.max():.2f}, "
                 f"and an SVD-based NNDSVDa start reached {INIT_NNDSVD[-1]:.2f}.", FadeIn(nmf[3]),
                 *self.cite("NNDSVD: Boutsidis & Gallopoulos 2008, Pattern Recognit."))
        self.say(f"Fourth, regularization. An L1 penalty with alpha = {SPARSE_ALPHA:g} raised the zeros in H "
                 f"from {100 * SPARSE_H0:.0f} to {100 * SPARSE_H1:.0f} percent, while the error rose from "
                 f"{SPARSE_ERR0:.2f} to {SPARSE_ERR1:.2f}.", FadeIn(nmf[4]),
                 *self.cite("sparseness: Hoyer 2004, JMLR"))
        self.say("Finally preprocessing, such as baseline removal and normalization, and the solver with its "
                 "stopping tolerance.", FadeIn(nmf[5]), FadeIn(nmf[6]))
        self.finish()


class S08Questions(NMFScene):
    level = 3
    part = P2
    speaker = "B"

    def construct(self):
        title = section_title("Biochemical questions NMF makes easier")
        cell = VGroup(Circle(radius=0.32, stroke_color="#A9DFBF", fill_color="#1E8449", fill_opacity=0.4),
                      Circle(radius=0.12, stroke_width=0, fill_color="#A9DFBF", fill_opacity=0.9))
        items = [
            (mini_spectrum(), "Which species are mixed in this spectrum?  (UV–vis, fluorescence, CD, Raman)"),
            (mini_kinetics(), "Is there a kinetic intermediate, and what does it look like?"),
            (tissue(), "Where is each metabolite or lipid in a tissue?  (mass-spectrometry imaging)"),
            (dna(n=8, length=1.2, amp=0.18, mutate=4), "Which DNA-damage processes shaped a genome?  (mutational signatures)"),
            (cell, "Which gene programs run in which cells?  (single-cell RNA-seq)"),
            (neuron(), "Which neurons fire, and when?  (calcium imaging, CNMF)"),
        ]
        rows = VGroup()
        for icon, text in items:
            fit_box(icon, 1.0, 0.6)
            ic = VGroup(icon).move_to(ORIGIN)
            box = Rectangle(width=1.1, height=0.65, stroke_width=0).move_to(ic)
            rows.add(VGroup(VGroup(box, ic), T(text, 24)).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        fit_box(rows, config.frame_width - 1.0, TOP - STAGE_BOTTOM - 0.1)
        rows.move_to([0, (TOP + STAGE_BOTTOM) / 2, 0])
        self.say("The same move answers many biochemical questions. Which species are mixed in a spectrum? "
                 "Is there a kinetic intermediate? Where is each metabolite in a tissue?",
                 Write(title), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows[:3]], lag_ratio=0.5),
                 run_time=3)
        self.say("Which DNA-damage processes shaped a genome? Which gene programs run in which cells? And which "
                 "neurons fire, and when? The fourth question is our paper.",
                 LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows[3:]], lag_ratio=0.5),
                 *self.cite("Alexandrov 2013 · Kotliar 2019 (cNMF) · Pnevmatikakis 2016 (CNMF)"), run_time=3)
        self.finish()


# ================================================================ PART 3
class S09Paper(NMFScene):
    level = 3
    part = P3

    def construct(self):
        cite = VGroup(T("Alexandrov et al. (2013)  Nature 500, 415–421", 30, color=YELLOW),
                      T("“Signatures of mutational processes in human cancer”", 28)
                      ).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.45)
        self.say("Our paper is Alexandrov and colleagues, Nature 2013: Signatures of mutational processes in "
                 "human cancer.", FadeIn(cite[0]), Write(cite[1]), run_time=2)
        cell = tumour_cell().scale(1.0)
        helix = dna(n=12, length=3.4, amp=0.32, mutate=6)
        g = int(np.argmax(MUT_E[:, 1] + MUT_E[:, 3]))
        prof0 = sig_bars(MUT_V[g], width=3.2, height=0.9, header=False)
        pipe = VGroup(cell, helix, prof0).arrange(RIGHT, buff=1.2).move_to([0, 0.7, 0])
        arrows = VGroup(Arrow(cell.get_right(), helix.get_left(), buff=0.15, color=GREY_B),
                        Arrow(helix.get_right(), prof0.get_left(), buff=0.15, color=GREY_B))
        plabs = VGroup(T("tumour", 20, color=GREY_A).next_to(cell, DOWN, buff=0.2),
                       T("sequenced DNA: one mutated base pair", 20, color=GREY_A).next_to(helix, DOWN, buff=0.35),
                       T("mutation counts", 20, color=GREY_A).next_to(prof0, DOWN, buff=0.35))
        self.say("Every cancer genome carries somatic mutations. Different processes, such as ultraviolet "
                 "light, tobacco smoke or enzymes that edit DNA, damage DNA in different ways and so leave "
                 "different patterns.", FadeIn(cell), GrowArrow(arrows[0]), Create(helix), FadeIn(plabs[:2]),
                 run_time=3)
        nums = T(f"{PAPER_MUTATIONS:,} mutations  ·  {PAPER_CANCERS:,} cancers ({PAPER_GENOMES} genomes + "
                 f"{PAPER_EXOMES:,} exomes)  ·  {PAPER_TYPES} cancer types", 24)
        fit_width(nums).move_to([0, -1.6, 0])
        self.say(f"The authors collected {PAPER_MUTATIONS:,} mutations from {PAPER_CANCERS:,} cancers: "
                 f"{PAPER_GENOMES} whole genomes and {PAPER_EXOMES:,} exomes, across {PAPER_TYPES} cancer types.",
                 GrowArrow(arrows[1]), FadeIn(prof0), FadeIn(plabs[2]), FadeIn(nums), run_time=2)

        self.clear_stage(cite, run_time=0.6)
        cls = M(r"6\ \text{substitutions} \times 4\ (5'\ \text{base}) \times 4\ (3'\ \text{base}) = 96\ \text{classes}",
                40).move_to([0, 1.7, 0])
        ex = T("e.g.  T[C>T]G  =  a C→T change with T on the 5′ side and G on the 3′ side", 24, color=GREY_A)
        ex.next_to(cls, DOWN, buff=0.3)
        self.say("Each substitution is classified by the base change and its two neighbours: six substitution "
                 "types, times four bases on the five-prime side, times four on the three-prime side, gives "
                 "96 classes.", Write(cls), FadeIn(ex), run_time=2.5)
        prof = sig_bars(MUT_V[g], width=10.5, height=1.4).move_to([0, -1.0, 0])
        vg = M(rf"\mathbf v_g \in \mathbb{{N}}^{{96}}\ \ (\text{{simulated genome}},\ {int(MUT_V[g].sum()):,}\ "
               r"\text{mutations})", 30).next_to(prof, DOWN, buff=0.15)
        self.say(f"So each genome becomes a vector of 96 counts; this simulated one has {int(MUT_V[g].sum()):,} "
                 "mutations. Stacked, the genomes form a non-negative count matrix: exactly NMF's input.",
                 FadeIn(prof[0]), FadeIn(prof[2:]), LaggedStart(*[GrowFromEdge(b, DOWN) for b in prof[1]],
                                                                 lag_ratio=0.01), Write(vg), run_time=2.5)
        self.finish()


class S10Method(NMFScene):
    level = 3
    part = P3

    def construct(self):
        title = section_title("How the paper used NMF")
        e1 = M(r"\underbrace{M}_{96\times G} \;\approx\; \underbrace{P}_{96\times k}\;\underbrace{E}_{k\times G}"
               r"\qquad\Longleftrightarrow\qquad \underbrace{V}_{G\times 96} = M^{\top} \approx "
               r"\underbrace{W}_{E^{\top}}\;\underbrace{H}_{P^{\top}}", 40).move_to([0, 1.9, 0])
        fit_width(e1)
        self.say("The paper writes the catalogue matrix M, mutation classes by genomes, as P times E: "
                 "signatures times exposures. In our notation V is M transposed, H holds the signatures and W "
                 "the exposures.", Write(title), Write(e1),
                 *self.cite("method: Alexandrov et al. 2013, Cell Reports 3, 246–259"), run_time=2.5)
        e2 = M(r"m_{jg} \;=\; \sum_{r=1}^{k} p_{jr}\; e_{rg}", 40).next_to(e1, DOWN, buff=0.45)
        self.say("Each count is a sum over processes: how often process r produces mutation class j, times how "
                 "strongly process r acted on genome g.", Write(e2))
        steps = bullet_list([
            f"NMF run separately in each of the {PAPER_TYPES} cancer types",
            "bootstrap resampling of the catalogues guards against over-fitting",
            "k chosen where signatures are reproducible AND reconstruction error is low",
            f"similar signatures clustered across cancer types → {PAPER_SIGNATURES} validated signatures",
        ], size=24, buff=0.16)
        ref = T("method: Alexandrov et al. (2013) Cell Reports 3, 246–259", 18, color=GREY_A)
        blk = VGroup(steps, ref).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        fit_width(blk).next_to(e2, DOWN, buff=0.4)
        self.say(f"They ran NMF separately in each of the {PAPER_TYPES} cancer types, with bootstrap "
                 "resampling to guard against over-fitting.", FadeIn(steps[0]), FadeIn(steps[1]))
        self.say(f"They chose k where the signatures were reproducible across runs and the reconstruction error "
                 f"was low, then clustered similar signatures across cancer types into {PAPER_SIGNATURES} "
                 "validated signatures.", FadeIn(steps[2]), FadeIn(steps[3]), FadeIn(ref))

        self.clear_stage(title, run_time=0.6)
        kr = MUT_K_RANGE
        e_grp, e_ax = small_axes([kr[0] - 0.5, kr[-1] + 0.5, 1], [0, 250, 50], 4.6, 2.5, kr, (0, 100, 200),
                                 "k", "reconstruction error (KL)")
        s_grp, s_ax = small_axes([kr[0] - 0.5, kr[-1] + 0.5, 1], [0, 1.05, 0.25], 4.6, 2.5, kr, (0, 0.5, 1),
                                 "k", "reproducibility: worst matched cosine, 6 restarts")
        VGroup(e_grp, s_grp).arrange(RIGHT, buff=1.4).move_to([0, -0.95, 0])
        e_ln = e_ax.plot_line_graph(kr, MUT_K_ERR, line_color=YELLOW, stroke_width=3,
                                    vertex_dot_style={"color": YELLOW}, vertex_dot_radius=0.06)
        s_ln = s_ax.plot_line_graph(kr, MUT_K_STAB, line_color=TEAL, stroke_width=3,
                                    vertex_dot_style={"color": TEAL}, vertex_dot_radius=0.06)
        mini = T(f"in miniature: {MUT_G} simulated genomes, {MUT_TOTAL:,} mutations, "
                 f"{MUT_SIG.shape[0]} planted processes, KL loss", 24).move_to([0, 2.6, 0])
        fit_width(mini)
        self.say(f"We reproduced the choice of k in miniature: {MUT_G} simulated genomes, {MUT_TOTAL:,} "
                 "mutations and four planted processes, fitted with the KL loss because the data are counts.",
                 FadeIn(mini), FadeIn(e_grp), Create(e_ln), run_time=2)
        cosm = M(r"\cos(\mathbf p, \hat{\mathbf p}) = \frac{\mathbf p \cdot \hat{\mathbf p}}"
                 r"{\lVert \mathbf p \rVert\,\lVert \hat{\mathbf p} \rVert}", 34).move_to([0, 1.75, 0])
        k_i = kr.index(4)
        marks = VGroup(Circle(radius=0.2, color=WHITE).move_to(e_ax.c2p(4, MUT_K_ERR[k_i])),
                       Circle(radius=0.2, color=WHITE).move_to(s_ax.c2p(4, MUT_K_STAB[k_i])))
        self.say(f"Reproducibility is the worst cosine similarity between matched signatures across six "
                 f"restarts. At k = 4 the error stops falling and reproducibility is still "
                 f"{MUT_K_STAB[k_i]:.3f}; at k = 5 it drops to {MUT_K_STAB[k_i + 1]:.3f}.",
                 Write(cosm), FadeIn(s_grp), Create(s_ln), Create(marks), run_time=2.5)
        self.say(f"With k = 4 the recovered signatures match the planted ones with cosine similarity "
                 f"{MUT_COS.min():.3f} to {MUT_COS.max():.3f}.", Indicate(marks))
        self.finish()


class S11KeyFigure(NMFScene):
    part = P3
    speaker = "B"
    level = 3

    def construct(self):
        title = section_title("The paper's Fig. 2: mutational signatures")
        notes = [
            ("C>T at NpCpG", "5-methylcytosine deamination; rises with age", clock()),
            ("C>T and C>G at TpCpN", "APOBEC cytidine deaminases", enzyme()),
            ("mostly C>A, strand-biased", "tobacco-smoke DNA adducts; lung", cigarette()),
            ("C>T after a pyrimidine", "UV light; melanoma", sun()),
        ]
        rows = VGroup()
        for i, (name, (patt, cause, icon)) in enumerate(zip(MUT_SIG_NAMES, notes)):
            prof = sig_bars(MUT_SIG_EST[i], width=7.0, height=0.85, header=(i == 0))
            fit_box(icon, 0.7, 0.7)
            txt = VGroup(T(name, 26, color=YELLOW), T(patt, 23), T(cause, 21, color=GREY_A)
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
            icon.next_to(prof[0], RIGHT, buff=0.45)
            txt.next_to(icon, RIGHT, buff=0.3)
            VGroup(icon, txt).align_to(prof[0], DOWN)
            icon.set_y(txt.get_y())
            rows.add(VGroup(prof, icon, txt))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        fit_box(rows, 12.8, TOP - STAGE_BOTTOM - 0.45)
        rows.move_to([0, (TOP + STAGE_BOTTOM) / 2 + 0.2, 0])
        foot = T("profiles: NMF output (rows of H) from our simulated catalogue, shaped after the paper's "
                 "descriptions, not the published values", 15, color=GREY_A)
        fit_width(foot, 12.4).next_to(rows, DOWN, buff=0.1)
        self.say("The paper's key Figure 2 draws each signature, a row of H, over the 96 classes. Ours come from "
                 "the simulation, shaped after the paper's descriptions.", Write(title), FadeIn(foot), run_time=1.5)
        lines = [
            "Signature 1 is C to T at CpG sites, from spontaneous deamination of 5-methylcytosine to thymine. "
            "It rises with patient age, like a clock.",
            "Signatures 2 and 13 are C to T and C to G changes at TpC sites, attributed to APOBEC cytidine "
            "deaminases, enzymes that edit DNA.",
            "Signature 4 is mostly C to A with a transcriptional strand bias; it is found in smoking-associated "
            "cancers such as lung, and fits bulky tobacco-smoke adducts on DNA.",
            "Signature 7 is C to T after a pyrimidine, the features of ultraviolet damage, and it is found in "
            "melanoma.",
        ]
        for r, text in zip(rows, lines):
            self.say(text, FadeIn(r[0], lag_ratio=0.01), FadeIn(r[1], scale=1.3), FadeIn(r[2], shift=LEFT * 0.15),
                     run_time=1.5)

        # Fig. 3-style panel from the NMF exposures W
        self.clear_stage(run_time=0.6)
        title = section_title("The paper's Fig. 3: which process acts where")
        cell = 0.95
        dots = VGroup()
        for g in range(4):
            for r in range(4):
                rad = 0.06 + 0.4 * np.sqrt(FIG3_MED[g, r])
                dots.add(Circle(radius=rad, stroke_width=0, fill_color=YELLOW, fill_opacity=0.85)
                         .move_to([r * cell, -g * cell, 0]))
        grid_lines = VGroup(*[Square(side_length=cell, stroke_color=GREY_D, stroke_width=1)
                              .move_to([r * cell, -g * cell, 0]) for g in range(4) for r in range(4)])
        cols = VGroup(*[T(n_.replace("Signatures ", "Sig. ").replace("Signature ", "Sig. "), 18)
                        .rotate(PI / 5).next_to(grid_lines[r], UP, buff=0.12) for r, n_ in enumerate(MUT_SIG_NAMES)])
        rows_l = VGroup(*[T(f"simulated {MUT_GROUP_NAMES[g]}", 18).next_to(grid_lines[4 * g], LEFT, buff=0.15)
                          for g in range(4)])
        key = T("dot area = median share of a genome's mutations (from W)", 17, color=GREY_A)
        key.next_to(grid_lines, DOWN, buff=0.2)
        fig = VGroup(grid_lines, dots, cols, rows_l, key)
        fit_box(fig, 6.4, 3.75).move_to([-3.3, 0.85, 0])
        pts = VGroup(
            T(f"the paper: every signature", 26), T(f"across {PAPER_TYPES} cancer types", 26),
            T("ours: the exposures W", 26, color=YELLOW), T("across 4 simulated groups", 26, color=YELLOW),
            T("each group lights up for the", 24, color=GREY_A), T("process planted in it;", 24, color=GREY_A),
            T("Signature 1 is everywhere", 24, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        for i in (2, 4):
            pts[i].shift(DOWN * 0.15)
            pts[i + 1:].shift(DOWN * 0.15)
        fit_box(pts, 6.0, 3.7).move_to([3.5, 0.85, 0])
        self.say(f"Figure 3 maps every signature across the {PAPER_TYPES} cancer types. Our miniature does the same "
                 "with the exposures W: each simulated group lights up for the process we planted, and Signature 1 "
                 "is everywhere.", Write(title), FadeIn(grid_lines), FadeIn(cols), FadeIn(rows_l),
                 LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.05), FadeIn(key),
                 FadeIn(pts, lag_ratio=0.3), run_time=3)
        box = VGroup(
            T("Insight: a tumour genome records the chemistry of the DNA damage and", 22, color=YELLOW),
            T(f"repair it went through; ~{PAPER_MUTATIONS / 1e6:.1f} M mutations → {PAPER_SIGNATURES} processes, "
              "many of unknown cause.", 22, color=YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        fit_width(box, 12.0)
        frame = SurroundingRectangle(box, color=YELLOW, buff=0.18)
        VGroup(box, frame).move_to([0, STAGE_BOTTOM + 0.5, 0])
        self.say("The biological insight: a tumour genome records the chemistry of the DNA damage and repair it "
                 f"experienced. NMF turned nearly five million mutations into {PAPER_SIGNATURES} interpretable "
                 "processes, and the many with unknown causes became new hypotheses.",
                 Create(frame), FadeIn(box), run_time=2)
        self.finish()


# ================================================================ PART 4
def panel_axes(x_range, y_range, w=5.8, h=2.0):
    return Axes(x_range=x_range, y_range=y_range, x_length=w, y_length=h, tips=False,
                axis_config={"include_numbers": False, "stroke_width": 1.5})


def wl_ticks(ax):
    return VGroup(*[T(str(x), 14).next_to(ax.c2p(x, 0), DOWN, buff=0.08) for x in (400, 500, 600, 700)])


class S12StrengthsLimits(NMFScene):
    part = P4
    level = 3

    def construct(self):
        title = section_title("Strengths")
        full, small = KIN_COMPRESS
        good = bullet_list([
            "physically meaningful parts",
            "model-free: no mechanism needed",
            f"compresses: {full:,} → {small:,} numbers",
            "KL loss handles counts",
            "fast; scales to huge matrices",
        ], size=32, buff=0.45, colors={"•": GREEN})
        fit_width(good, 6.4).move_to([-3.6, 0.2, 0])
        # before -> after thumbnail of the kinetics example
        a1 = panel_axes([400, 700, 100], [0, 1.1, 0.5], 2.5, 2.6)
        a2 = panel_axes([400, 700, 100], [0, 1.1, 0.5], 2.5, 2.6)
        raw = VGroup(*[curve(a1, WL, KIN_X[i], interpolate_color(BLUE_A, PINK, i / 40), 1.2, 0.8)
                       for i in range(0, KIN_T.size, 2)])
        rec = VGroup(*[curve(a2, WL, KIN_S_EST[c], COMP_COLORS[c], 3) for c in range(3)])
        g1, g2 = VGroup(a1, raw), VGroup(a2, rec)
        thumbs = VGroup(g1, g2).arrange(RIGHT, buff=0.8).move_to([3.55, 0.4, 0])
        arr = Arrow(g1.get_right(), g2.get_left(), buff=0.1, color=YELLOW)
        tl = VGroup(T(f"before: {KIN_X.shape[0]} raw spectra", 19, color=GREY_A).next_to(g1, DOWN, buff=0.15),
                    T("after: 3 pure spectra", 19, color=GREY_A).next_to(g2, DOWN, buff=0.15))
        self.say("What is NMF good at? Its parts are physically meaningful, and it is model-free: we never told it "
                 "the mechanism or gave it reference spectra.", Write(title), FadeIn(good[0]), FadeIn(good[1]),
                 FadeIn(g1), FadeIn(tl[0]), run_time=1.5)
        self.say(f"It compresses and denoises, {full:,} numbers to {small:,} in the kinetics example; with the KL "
                 "loss it handles counts; and it scales to very large matrices.",
                 FadeIn(good[2:], lag_ratio=0.3), GrowArrow(arr), FadeIn(g2), FadeIn(tl[1]), run_time=2)

        # failure modes: four computed panels
        self.clear_stage(run_time=0.6)
        title = section_title("Failure modes, with evidence")
        fin = INIT_RANDOM[:, -1]

        def panel(head, body, ax, note):
            h = T(head, 26, color=RED_B)
            n = T(note, 18, color=GREY_A)
            fit_width(n, 6.3)
            g = VGroup(h, body, n).arrange(DOWN, buff=0.12)
            return g

        # (1) not unique
        ax1 = panel_axes([400, 700, 100], [0, 1.1, 0.5])
        b1 = VGroup(ax1, wl_ticks(ax1),
                    *[DashedVMobject(curve(ax1, WL, ROT_S[c], COMP_COLORS[c], 2.5), num_dashes=40) for c in range(2)],
                    *[curve(ax1, WL, ROT_H[c], COMP_COLORS[c], 3.5) for c in range(2)])
        p1 = panel("1  not unique", b1, ax1,
                   f"dashed = truth (peaks {ROT_PEAKS_TRUE[0]:.0f}, {ROT_PEAKS_TRUE[1]:.0f} nm), solid = NMF "
                   f"({ROT_PEAKS_NMF[0]:.0f}, {ROT_PEAKS_NMF[1]:.0f} nm); both fit")
        # (2) k too large
        ax2 = panel_axes([400, 700, 100], [0, 1.1, 0.5])
        b2 = VGroup(ax2, wl_ticks(ax2), *[curve(ax2, WL, SPLIT_H[r], COMP_COLORS[SPLIT_MATCH[r]], 3, 0.9)
                                          for r in range(SPLIT_H.shape[0])])
        p2 = panel("2  k too large", b2, ax2,
                   f"k = 4 on 3 species: green split in two (stability {K_STAB[2]:.3f} → {K_STAB[3]:.3f})")
        # (3) start-dependent
        ax3 = panel_axes([0, INIT_ITERS, 100], [0, 3.2, 1])
        it = np.arange(INIT_ITERS + 1)
        b3 = VGroup(ax3, VGroup(*[T(str(x), 14).next_to(ax3.c2p(x, 0), DOWN, buff=0.08) for x in (0, 100, 200, 300)]),
                    *[curve(ax3, it, np.log10(L), GREY_B, 1.5, 0.8) for L in INIT_RANDOM],
                    curve(ax3, it, np.log10(INIT_NNDSVD), YELLOW, 3))
        p3 = panel("3  depends on the start", b3, ax3,
                   f"log loss vs iteration: 6 random starts end at {fin.min():.2f}–{fin.max():.2f}; "
                   f"NNDSVDa (yellow) {INIT_NNDSVD[-1]:.2f}")
        # (4) baseline offset
        ax4 = panel_axes([400, 700, 100], [0, 1.1, 0.5])
        b4 = VGroup(ax4, wl_ticks(ax4),
                    *[DashedVMobject(curve(ax4, WL, S_TRUE[c], COMP_COLORS[c], 2), num_dashes=40) for c in range(3)],
                    *[curve(ax4, WL, BASE_H[c], COMP_COLORS[c], 3) for c in range(3)])
        p4 = panel("4  model violations", b4, ax4,
                   f"+{BASE_OFFSET:g} baseline on every spectrum: the parts gain a floor of up to {BASE_FLOOR:.2f}")
        grid_ = VGroup(p1, p2, p3, p4).arrange_in_grid(rows=2, cols=2, buff=(0.6, 0.3))
        fit_box(grid_, 13.2, TOP - STAGE_BOTTOM - 0.05)
        grid_.move_to([0, (TOP + STAGE_BOTTOM) / 2 + 0.05, 0])
        self.say("Now the failure modes, each with evidence. First, the answer is not unique. With overlapping "
                 "bands and no pure samples, NMF's spectra differ from the truth, yet fit the data just as well.",
                 Write(title), FadeIn(p1), run_time=2, who="B")
        self.say(f"Second, k. Asked for four parts from three species, NMF splits the green species in two, and "
                 f"stability drops from {K_STAB[2]:.3f} to {K_STAB[3]:.3f}.", FadeIn(p2), who="B")
        self.say(f"Third, the result depends on the start: six random starts end between {fin.min():.2f} and "
                 f"{fin.max():.2f}.", FadeIn(p3), who="B")
        self.say("Fourth, real data break the model. A constant baseline is not an additive part, so it leaks into "
                 "the recovered spectra; scattering, peak shifts and a wrong noise model do the same.",
                 FadeIn(p4), who="B")
        self.finish()


class S13Theory(NMFScene):
    part = P4
    level = 4
    speaker = "B"

    def construct(self):
        title = section_title("Graduate level: uniqueness and complexity")
        amb = M(r"WH = (WT)(T^{-1}H)\quad\text{for every invertible } T \text{ with } WT \ge 0,\ T^{-1}H \ge 0", 34)
        fit_width(amb).move_to([0, 2.45, 0])
        self.say("At graduate level the questions are identifiability and complexity. Any invertible T that keeps "
                 "both factors non-negative gives the same product, so when is the answer unique?",
                 Write(title), Write(amb), run_time=2)

        def simplex_panel(P, label, inner=None):
            sc = 1.25
            tri = Polygon(*[[*(v * sc), 0] for v in SIMPLEX_V], stroke_color=WHITE, stroke_width=2.5)
            pts = VGroup(*[Dot([*(p * sc), 0], radius=0.025, color=GREY_B) for p in P])
            g = VGroup(tri, pts)
            names = VGroup(*[M(rf"\mathbf h_{i + 1}", 26, color=COMP_COLORS[i]).next_to(
                [*(SIMPLEX_V[i] * sc), 0], [UP, DL, DR][i], buff=0.08) for i in range(3)])
            g.add(names)
            if inner is not None:
                g.add(DashedVMobject(Polygon(*[[*(v * sc), 0] for v in inner], stroke_color=YELLOW,
                                             stroke_width=2.5), num_dashes=36))
            lab = VGroup(*[T(x, 20, color=GREY_A) for x in label]).arrange(DOWN, buff=0.06)
            lab.next_to(tri, DOWN, buff=0.4)
            return VGroup(g, lab)

        pa = simplex_panel(SIMPLEX_SPREAD_XY, ["samples reach the edges:", "one triangle fits (unique)"])
        pb = simplex_panel(SIMPLEX_CLUSTER_XY, ["samples bunched:", "other triangles fit too"], SIMPLEX_INNER)
        panels = VGroup(pa, pb).arrange(RIGHT, buff=0.7)
        fit_box(panels, 6.8, 4.6).move_to([-3.35, -0.05, 0])
        self.say("Normalise each row of W and every sample becomes a point in a triangle whose corners are the "
                 "pure parts. If the samples reach the edges, only one triangle fits, so the factorization is "
                 "unique. If they are bunched in the middle, many triangles fit, like the dashed one.",
                 FadeIn(pa), FadeIn(pb), run_time=2.5)
        items = [
            ("exact NMF is NP-hard in general", "Vavasis 2009"),
            ("separable (a pure sample per part):", "provably solvable · Arora et al. 2012"),
            ("'sufficiently scattered' rows of W:", "unique · Huang et al. 2014; Fu et al. 2019"),
            ("faster solvers: ANLS, HALS", "Kim & Park 2008; Cichocki & Phan 2009"),
            ("general losses: β-divergences", "Févotte & Idier 2011"),
        ]
        facts = VGroup(*[VGroup(T(f"•  {a}", 24), T(b, 19, color=GREY_A)).arrange(DOWN, aligned_edge=LEFT,
                                                                                  buff=0.06) for a, b in items])
        for f in facts:
            f[1].shift(RIGHT * 0.38)
        facts.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        fit_box(facts, 6.4, 4.7).move_to([3.6, -0.05, 0])
        self.say("In general, exact NMF is NP-hard, as Vavasis showed. But if every part has a pure sample, the "
                 "separable case, Arora and colleagues gave a provably correct algorithm, and Huang, Sidiropoulos "
                 "and Fu proved uniqueness under the weaker sufficiently scattered condition.",
                 FadeIn(facts[:3], lag_ratio=0.4), run_time=2.5)
        self.say("In practice, alternating non-negative least squares and HALS converge faster than multiplicative "
                 "updates, and beta-divergences generalize the loss.", FadeIn(facts[3:], lag_ratio=0.4))
        self.finish()


class S14Conclusions(NMFScene):
    part = P4
    level = 4

    def construct(self):
        title = section_title("Conclusions NOT justified by W and H alone")
        claims = [
            ("“Component B is one pure species.”", "it could be several species that always change together"),
            ("“B reaches 0.6 mM.”", "W has no absolute scale, (WD)(D⁻¹H) fits equally well; needs ε or a calibration"),
            ("“There are exactly 3 species.”", "k is a modelling choice, not a measurement"),
            ("“Signature 4 is caused by smoking.”", "NMF finds a pattern; the cause needs outside evidence"),
        ]
        rows = VGroup(*[VGroup(T(f"✗  {c}", 32, color=RED_B), T(w, 25, color=GREY_A)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12) for c, w in claims])
        for r in rows:
            r[1].shift(RIGHT * 0.55)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        fit_box(rows, 12.0, TOP - STAGE_BOTTOM - 0.2).move_to([0, (TOP + STAGE_BOTTOM) / 2, 0])
        self.say("Which conclusions would not be justified from the transformed data alone? Component B is one pure "
                 "species: it could be several species that always change together.",
                 Write(title), FadeIn(rows[0]), run_time=1.5)
        self.say("B reaches 0.6 millimolar: W has no absolute scale, so that needs an extinction coefficient.",
                 FadeIn(rows[1]))
        self.say("There are exactly three species: k was our choice. And Signature 4 is caused by smoking: NMF "
                 "finds a pattern, and the cause needs outside evidence.", FadeIn(rows[2]), FadeIn(rows[3]))

        self.clear_stage(run_time=0.6)
        h = section_title("The four questions, answered")
        qa = [
            ("What goes in?", M(r"V \in \mathbb{R}_{\ge0}^{\,n\times m}\ \ (\text{samples} \times \text{features})", 36)),
            ("What comes out?", M(r"W \in \mathbb{R}_{\ge0}^{\,n\times k}\ (\text{amounts}),\ \ "
                                  r"H \in \mathbb{R}_{\ge0}^{\,k\times m}\ (\text{parts}),\ \ V \approx WH", 36)),
            ("Kept / lost?", T("kept: additive, low-rank structure;  lost: noise, scale, order, minor parts", 26)),
            ("Easier question?", T("which pure components are mixed here, and how much of each?", 26)),
        ]
        tab = VGroup(*[VGroup(T(q, 28, color=YELLOW), a) for q, a in qa])
        qw = max(r[0].width for r in tab)
        for r in tab:
            r[1].next_to(r[0], RIGHT, buff=0.45 + qw - r[0].width)
        tab.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        fit_box(tab, 12.6, TOP - STAGE_BOTTOM - 0.3).move_to([0, (TOP + STAGE_BOTTOM) / 2, 0])
        self.say("To answer the four questions. What goes in: a non-negative matrix, samples by features. What comes "
                 "out: W, the amounts, and H, the parts.", Write(h), FadeIn(tab[0]), FadeIn(tab[1]),
                 run_time=2)
        self.say("What is kept is the additive structure; what is lost is noise, scale, order and minor components. "
                 "And the question that gets easier: which pure components are mixed here, and how much of each?",
                 FadeIn(tab[2]), FadeIn(tab[3]))

        # reading path, beginner -> graduate
        self.clear_stage(run_time=0.6)
        h2 = section_title("Further reading: beginner → graduate")
        cols = [
            ("beginner", LEVELS[0][1], ["Lee & Seung 1999, Nature", "Gillis 2014, “The why and how of NMF”",
                                         "Brunet et al. 2004, PNAS"]),
            ("intermediate", LEVELS[1][1], ["Paatero & Tapper 1994, Environmetrics", "Lee & Seung 2001, NIPS",
                                             "de Juan & Tauler 2006, MCR review", "Alexandrov et al. 2013, Nature",
                                             "Kotliar et al. 2019, eLife (cNMF)"]),
            ("advanced", LEVELS[2][1], ["Hoyer 2004, JMLR (sparsity)", "Lin 2007, Neural Comput.",
                                         "Kim & Park 2008, SIMAX (ANLS)", "Cichocki & Phan 2009 (HALS)",
                                         "Févotte & Idier 2011 (β-div.)"]),
            ("graduate", LEVELS[3][1], ["Donoho & Stodden 2003, NIPS", "Vavasis 2009, SIAM J. Optim.",
                                         "Arora et al. 2012, STOC", "Huang et al. 2014, IEEE TSP",
                                         "Fu et al. 2019, IEEE SPM", "Gillis 2020, SIAM book"]),
        ]
        cg = VGroup()
        for name, col, refs in cols:
            cg.add(VGroup(T(name, 30, color=col), *[T(r_, 22) for r_ in refs]).arrange(DOWN, aligned_edge=LEFT,
                                                                                       buff=0.2))
        cg.arrange(RIGHT, aligned_edge=UP, buff=0.5)
        fit_box(cg, 13.3, 4.0).move_to([0, 0.75, 0])
        arrow = Arrow(cg.get_left() + DOWN * 0.1, cg.get_right() + DOWN * 0.1, buff=0, color=GREY_B,
                      stroke_width=3).next_to(cg, DOWN, buff=0.25)
        end = VGroup(VWH(44), T("Thank you! Questions?", 28, color=GREY_A)).arrange(RIGHT, buff=0.6)
        end.next_to(arrow, DOWN, buff=0.25)
        self.say("For further reading, from beginner to graduate level: start with Lee and Seung's Nature paper and "
                 "Gillis's tutorial; then the algorithms and the chemical and biological applications; and finally "
                 "the theory of uniqueness and complexity. Full references are in our README.",
                 Write(h2), LaggedStart(*[FadeIn(c, shift=UP * 0.2) for c in cg], lag_ratio=0.4), GrowArrow(arrow),
                 run_time=4, who="B")
        self.say("Thank you. We are happy to take questions.", FadeIn(end, shift=UP * 0.2), who="B")
        self.finish()
