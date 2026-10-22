"""Small vector illustrations drawn with Manim primitives (no image files).

Each function returns a VGroup centred on the origin; position/scale it in
the scene.
"""
import numpy as np
from manim import (BLACK, BLUE_E, DOWN, GREY_A, GREY_B, GREY_D, GREY_E, LEFT, ORANGE, ORIGIN, PI, RED,
                   RIGHT, UP, WHITE, YELLOW, Arc, Circle, Dot, Ellipse, Line, Polygon,
                   Rectangle, RoundedRectangle, Sector, VGroup, VMobject, interpolate_color)

from helpers import COMP_COLORS, T


def _molecules(width, height, n=14, seed=0, colors=COMP_COLORS):
    rng = np.random.default_rng(seed)
    return VGroup(*[Dot([rng.uniform(-width / 2, width / 2), rng.uniform(-height / 2, height / 2), 0],
                        radius=0.06, color=colors[i % len(colors)]) for i in range(n)])


def cuvette(colors=COMP_COLORS, n=14, seed=0):
    glass = Rectangle(width=0.9, height=1.6, stroke_color=GREY_A, stroke_width=3)
    liquid = Rectangle(width=0.84, height=1.15, stroke_width=0, fill_color=BLUE_E, fill_opacity=0.35)
    liquid.align_to(glass, DOWN).shift(UP * 0.03)
    mols = _molecules(0.7, 1.0, n, seed, colors).move_to(liquid)
    return VGroup(glass, liquid, mols)


def lamp():
    bulb = Circle(radius=0.3, stroke_width=0, fill_color=YELLOW, fill_opacity=0.95)
    rays = VGroup(*[Line(0.38 * np.array([np.cos(a), np.sin(a), 0]), 0.55 * np.array([np.cos(a), np.sin(a), 0]),
                         color=YELLOW, stroke_width=3) for a in np.linspace(0, 2 * PI, 9)[:-1]])
    return VGroup(bulb, rays)


def detector():
    box = RoundedRectangle(corner_radius=0.08, width=0.7, height=1.0, stroke_color=GREY_A,
                           fill_color=GREY_D, fill_opacity=1)
    eye = Rectangle(width=0.08, height=0.5, stroke_width=0, fill_color=WHITE, fill_opacity=0.9).move_to(box.get_left() + RIGHT * 0.1)
    return VGroup(box, eye)


def spectrometer(colors=COMP_COLORS, labels=True):
    """lamp -> beam -> cuvette with mixed molecules -> detector"""
    L, C, D = lamp(), cuvette(colors), detector()
    L.move_to(LEFT * 2.6)
    D.move_to(RIGHT * 2.6)
    beam1 = Line(L.get_right(), C.get_left() + DOWN * 0.15, color=YELLOW, stroke_width=6)
    beam2 = Line(C.get_right() + DOWN * 0.15, D.get_left(), color=YELLOW, stroke_width=3, stroke_opacity=0.6)
    C.shift(UP * 0.15)
    g = VGroup(beam1, beam2, L, C, D)
    if labels:
        g.add(VGroup(T("lamp", 18, color=GREY_A).next_to(L, DOWN, buff=0.2),
                     T("mixture", 18, color=GREY_A).next_to(C, DOWN, buff=0.15),
                     T("detector", 18, color=GREY_A).next_to(D, DOWN, buff=0.2)))
    return g


def syringe(color):
    barrel = Rectangle(width=1.2, height=0.36, stroke_color=GREY_A, fill_color=color, fill_opacity=0.35)
    plunger = VGroup(Line(barrel.get_left() + RIGHT * 0.15 + UP * 0.16, barrel.get_left() + RIGHT * 0.15 + DOWN * 0.16,
                          stroke_width=4, color=GREY_A),
                     Line(barrel.get_left() + RIGHT * 0.15, barrel.get_left() + LEFT * 0.45, stroke_width=4, color=GREY_A))
    tip = Line(barrel.get_right(), barrel.get_right() + RIGHT * 0.2, stroke_width=4, color=GREY_A)
    return VGroup(barrel, plunger, tip)


def stopped_flow():
    """two syringes -> mixer -> observation cell in a light beam"""
    s1, s2 = syringe(COMP_COLORS[0]), syringe(GREY_B)
    s1.move_to(LEFT * 2.6 + UP * 0.7)
    s2.move_to(LEFT * 2.6 + DOWN * 0.7)
    mixer = Circle(radius=0.22, stroke_color=GREY_A, fill_color=GREY_E, fill_opacity=1).move_to(LEFT * 0.9)
    p1 = Line(s1[2].get_end(), mixer.get_left() + UP * 0.1, stroke_width=4, color=GREY_A)
    p2 = Line(s2[2].get_end(), mixer.get_left() + DOWN * 0.1, stroke_width=4, color=GREY_A)
    cell = cuvette(COMP_COLORS, n=12, seed=3).scale(0.75).move_to(RIGHT * 0.6)
    p3 = Line(mixer.get_right(), cell.get_left(), stroke_width=4, color=GREY_A)
    L = lamp().scale(0.6).move_to(RIGHT * 0.6 + UP * 1.6)
    D = detector().scale(0.7).rotate(PI / 2).move_to(RIGHT * 0.6 + DOWN * 1.55)
    beam = Line(L.get_bottom(), D.get_top(), color=YELLOW, stroke_width=4, stroke_opacity=0.7)
    labs = VGroup(T("A", 20, color=COMP_COLORS[0]).next_to(s1, UP, buff=0.08),
                  T("buffer", 18, color=GREY_A).next_to(s2, DOWN, buff=0.08),
                  T("mixer", 16, color=GREY_A).next_to(mixer, UP, buff=0.1),
                  T("spectrum every 2.5 s", 16, color=GREY_A).next_to(D, RIGHT, buff=0.15))
    return VGroup(beam, s1, s2, p1, p2, p3, mixer, cell, L, D, labs)


def dna(n=12, length=4.0, amp=0.35, mutate=6):
    """double helix; rung `mutate` drawn as a red mutated base pair"""
    xs = np.linspace(-length / 2, length / 2, 200)
    s1 = VMobject().set_points_smoothly([[x, amp * np.sin(2 * PI * x / 1.6), 0] for x in xs[::6]])
    s2 = VMobject().set_points_smoothly([[x, -amp * np.sin(2 * PI * x / 1.6), 0] for x in xs[::6]])
    s1.set_stroke(GREY_A, 4)
    s2.set_stroke(GREY_B, 4)
    rungs = VGroup()
    base_cols = ["#5DADE2", "#F5B041", "#58D68D", "#EC7063"]
    for i, x in enumerate(np.linspace(-length / 2 + 0.15, length / 2 - 0.15, n)):
        y = amp * np.sin(2 * PI * x / 1.6)
        r = Line([x, y, 0], [x, -y, 0], stroke_width=4, color=base_cols[i % 4])
        if i == mutate:
            r.set_stroke(RED, 8)
        rungs.add(r)
    return VGroup(rungs, s1, s2)


def tumour_cell(seed=1):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * PI, 14)[:-1]
    rad = 0.8 + rng.uniform(-0.12, 0.12, ang.size)
    body = VMobject().set_points_smoothly([[r * np.cos(a), r * np.sin(a), 0] for r, a in zip(rad, ang)] +
                                         [[rad[0], 0, 0]])
    body.set_fill("#C0392B", 0.35).set_stroke("#E6B0AA", 3)
    nucleus = Circle(radius=0.32, stroke_color="#F5B7B1", fill_color="#922B21", fill_opacity=0.8).shift(RIGHT * 0.1)
    return VGroup(body, nucleus)


def clock():
    face = Circle(radius=0.35, stroke_color=WHITE, stroke_width=3)
    hands = VGroup(Line(ORIGIN, UP * 0.24, stroke_width=3), Line(ORIGIN, RIGHT * 0.17, stroke_width=3))
    return VGroup(face, hands)


def enzyme():
    """Pac-Man-like enzyme biting a C"""
    body = Sector(radius=0.36, angle=1.6 * PI, start_angle=0.2 * PI, fill_color="#AF7AC5", fill_opacity=0.9,
                  stroke_width=0)
    base = T("C", 18, color=WHITE).move_to(RIGHT * 0.32)
    return VGroup(body, base)


def cigarette():
    body = Rectangle(width=0.9, height=0.16, stroke_width=0, fill_color=WHITE, fill_opacity=1)
    filt = Rectangle(width=0.25, height=0.16, stroke_width=0, fill_color=ORANGE, fill_opacity=1).align_to(body, LEFT)
    ember = Rectangle(width=0.06, height=0.16, stroke_width=0, fill_color=RED, fill_opacity=1).next_to(body, RIGHT, buff=0)
    smoke = VMobject().set_points_smoothly([[0.5, 0.12, 0], [0.6, 0.3, 0], [0.5, 0.45, 0], [0.62, 0.62, 0]])
    smoke.set_stroke(GREY_B, 2)
    return VGroup(body, filt, ember, smoke).center()


def sun():
    core = Circle(radius=0.2, stroke_width=0, fill_color=YELLOW, fill_opacity=1)
    rays = VGroup(*[Line(0.27 * np.array([np.cos(a), np.sin(a), 0]), 0.4 * np.array([np.cos(a), np.sin(a), 0]),
                         stroke_width=3, color=YELLOW) for a in np.linspace(0, 2 * PI, 11)[:-1]])
    return VGroup(core, rays)


def tissue():
    slab = Ellipse(width=1.0, height=0.6, stroke_color="#F1948A", fill_color="#78281F", fill_opacity=0.6)
    spots = VGroup(*[Dot(p, radius=0.06, color=c) for p, c in
                     [((-0.25, 0.05, 0), COMP_COLORS[0]), ((0.1, -0.1, 0), COMP_COLORS[1]), ((0.28, 0.12, 0), COMP_COLORS[2])]])
    return VGroup(slab, spots)


def neuron():
    soma = Circle(radius=0.16, stroke_width=0, fill_color="#5DADE2", fill_opacity=1)
    dend = VGroup(*[Line(0.16 * np.array([np.cos(a), np.sin(a), 0]), 0.42 * np.array([np.cos(a), np.sin(a), 0]),
                         stroke_width=2, color="#5DADE2") for a in (0.5, 1.6, 2.6, 3.6, 4.6)])
    axon = Line(RIGHT * 0.16, RIGHT * 0.55 + DOWN * 0.2, stroke_width=2, color="#5DADE2")
    return VGroup(soma, dend, axon)


def mini_spectrum():
    xs = np.linspace(-0.5, 0.5, 40)
    ys = 0.35 * np.exp(-((xs + 0.15) / 0.12) ** 2) + 0.25 * np.exp(-((xs - 0.2) / 0.1) ** 2)
    c = VMobject().set_points_smoothly([[x, y - 0.2, 0] for x, y in zip(xs, ys)]).set_stroke(GREY_A, 3)
    return VGroup(Line([-0.55, -0.2, 0], [0.55, -0.2, 0], stroke_width=2, color=GREY_B), c)


def mini_kinetics():
    ts = np.linspace(0, 1, 40)
    a = np.exp(-6 * ts)
    b = 6 / 2 * (np.exp(-2 * ts) - np.exp(-6 * ts)) * 0.8
    g = VGroup(Line([-0.5, -0.3, 0], [0.5, -0.3, 0], stroke_width=2, color=GREY_B))
    for y, col in ((a, COMP_COLORS[0]), (b, COMP_COLORS[1]), (1 - a - b / 0.8 * 0.8, COMP_COLORS[2])):
        g.add(VMobject().set_points_smoothly([[t - 0.5, 0.55 * v - 0.3, 0] for t, v in zip(ts, y)]).set_stroke(col, 3))
    return g
