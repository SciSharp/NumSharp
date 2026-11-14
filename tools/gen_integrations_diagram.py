"""Generates docs/website-src/images/integrations.svg: NumSharp in the middle, a neon data stream out to every integration.

The README's "Integrations" section and the top of the docs' Interoperability page (docs/website-src/docs/interop/index.md)
show this picture. Each integration sits in an octagon with the icon of its ecosystem, its name and one line that says how
an NDArray crosses into it; a dotted cable runs from NumSharp to it, and a burst of light runs out along every cable.

What belongs in the picture: an integration that is RELEASED (its package is published), DOCUMENTED (the Interoperability
pages describe it) and GATED (a test or an example's asserted run exercises it). A merged but unreleased bridge stays out,
exactly as it stays out of the README and the site (.claude/rules/interop-packages.md, "Unreleased packages"): the reader
must be able to install everything the picture shows. The color of an octagon names the package that brings it:

    Python  NumSharp.Interop.pythonnet (NumPy, PyTorch, pandas, SciPy, scikit-learn, Matplotlib, Pillow, OpenCV, PyArrow,
            polars; the Python libraries cross through numpy arrays or the buffer protocol)
    .NET    NumSharp.Interop.OnnxRuntime / .MLNet / .System.Numerics.Tensors / .TorchSharp, NumSharp.Bitmap
            (System.Drawing), and NumSharp.Interop.pythonnet for Numpy.NET
    Native  NumSharp.Interop.OpenBLAS
    Files   NumSharp itself (.npy / .npz, byte-identical to np.save)

How the picture is built. Everything sits on a character grid of 10 x 20 px cells, so it reads as the ASCII diagram it
started as: the octagons follow the cells' center lines with 45-degree chamfers, the NUMSHARP wordmark is the figlet
"future" letters drawn as heavy box-drawing strokes, and the captions are placed one character per cell. GitHub shows a
README image through an <img> tag, which loads no web fonts, so every text character gets its own x position: the
layout holds in whatever monospace font the reader's system substitutes. The NumSharp logo is read from
docs/website-src/images/numsharp.icon.svg at generation time (its Inkscape metadata stripped), so the picture always
carries the current logo.

Animation: each cable is a path of zero-length round-capped dashes (the dots) plus a second path whose dash pattern is a
short group of bright dots followed by a gap longer than the cable; a CSS animation moves that pattern outward, so about
fifty elements animate instead of one per dot. Readers who ask for reduced motion (prefers-reduced-motion) see the still
picture.

The docs home page (docs/website-src/index.md) shows the picture too, with a layer of links over it: hovering an octagon
(or tabbing to it) rings it in white and opens two buttons in its color under it, "Open Library" (the library's own site)
and "Open Docs" (its NumSharp docs page, which a click on the octagon itself also opens), and NumSharp's own octagon offers
the repository and the Interoperability guide. An <img> cannot carry links, so that layer is plain HTML positioned in
percentages of the picture; the script writes it between the INTEGRATIONS_LINKS markers of the home page from the same
layout that draws the octagons, so a box can never drift away from the octagon it covers. The links come from each
item's docs and site fields; DocFX resolves the .md paths and reports a page that disappears.

To change the picture, edit ITEMS (what an octagon says and where its buttons go) and SIDES (where it sits; each side
lists its octagons in reading order and the four sides keep each ecosystem on one arc of the ring), then run the script
and commit both the SVG and the home page. The script refuses a layout whose octagons overlap or leave the canvas, and a
home page whose markers are missing or repeated.

Run with any Python 3.8 or newer (standard library only):
    python tools/gen_integrations_diagram.py            write the SVG and the home page's link layer
    python tools/gen_integrations_diagram.py --check    exit 1 when the committed SVG or link layer differs from a fresh
                                                        render
"""
import html
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "docs", "website-src", "images", "integrations.svg")
LOGO = os.path.join(ROOT, "docs", "website-src", "images", "numsharp.icon.svg")
HOME = os.path.join(ROOT, "docs", "website-src", "index.md")

# The home page's link layer lives between these two lines of index.md; everything between them is rewritten on each run.
# Each marker must stand on a line of its own, exactly once, inside the picture's container (see render_links()).
INTEGRATIONS_LINKS_BEGIN = "<!-- integrations-links:begin -->"
INTEGRATIONS_LINKS_END = "<!-- integrations-links:end -->"

# ------------------------------------------------------------------------------------------------------------------------
# Geometry and type
# ------------------------------------------------------------------------------------------------------------------------

CW, CH = 10, 20      # one character cell in px; monospace glyphs are about 1:2, so a cell is too
PAD = 22             # margin around the grid, inside the rounded card
COLS, RING_ROWS = 128, 42   # the ring of octagons; two legend rows and a blank row follow it
SAT_ROWS = 4         # an octagon is four rows: top edge, name, caption, bottom edge

CENTER_COLS, CENTER_ROWS = 47, 11  # NumSharp's octagon: the logo and the 31-cell wordmark side by side, tagline below
LOGO_PX = 92                      # the logo's square, a little taller than the wordmark and tagline together

NAME_PX, NAME_ADV = 21, 12.6      # an integration's name: bold, 0.6 em per character
CAPTION_PX = 14.5                 # its caption: one character per cell
ICON_PX, ICON_GAP = 22, 8         # the ecosystem icon in front of the name

FONT = ("'JetBrains Mono','Cascadia Mono','SF Mono',Menlo,Consolas,'DejaVu Sans Mono','Liberation Mono',monospace")

NUMSHARP = "#B47CFF"        # the core's neon purple (NumSharp's logo is purple)
NUMSHARP_TEXT = "#D8C2FF"   # the wordmark, a paler purple so it reads over the glow

# ------------------------------------------------------------------------------------------------------------------------
# What the picture shows
# ------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One integration's octagon.

    Attributes:
        key: The identifier SIDES places it by.
        name: The library's own spelling, shown large.
        caption: How an NDArray crosses into it, in at most about 25 characters (a longer one widens the octagon and can
            push it into its neighbours, which the layout check then refuses).
        eco: The ecosystem key into ECOSYSTEMS; it decides the color, the icon and the legend entry.
        docs: The NumSharp docs page for it, relative to docs/website-src, as the home page links it ("Open Docs" and a
            click on the octagon). A .md path, optionally with a #heading anchor: DocFX turns it into the .html address
            and warns when the page is gone, which a hard-coded .html link would hide.
        site: The library's own website ("Open Library"); DocFX opens it in a new tab with its external-link mark.
    """
    key: str
    name: str
    caption: str
    eco: str
    docs: str
    site: str


ITEMS = [
    Item("numpy", "NumPy", "zero-copy, both ways", "py",
         "docs/interop/pythonnet-numpy.md", "https://numpy.org/"),
    Item("torch", "PyTorch", "zero-copy CPU tensors", "py",
         "docs/interop/pytorch.md", "https://pytorch.org/"),
    Item("pandas", "pandas", "DataFrame · Series", "py",
         "docs/interop/pandas.md", "https://pandas.pydata.org/"),
    Item("scipy", "SciPy", "signal · optimize", "py",
         "docs/interop/pythonnet-numpy.md", "https://scipy.org/"),
    Item("sklearn", "scikit-learn", "fit · predict", "py",
         "docs/interop/pythonnet-numpy.md", "https://scikit-learn.org/"),
    Item("mpl", "Matplotlib", "plots · pixel buffer", "py",
         "docs/interop/pythonnet-numpy.md", "https://matplotlib.org/"),
    Item("pillow", "Pillow", "Image.frombuffer", "py",
         "docs/interop/np-frombuffer.md#pillow", "https://python-pillow.org/"),
    Item("opencv", "OpenCV", "draws in place", "py",
         "docs/interop/np-frombuffer.md#opencv", "https://opencv.org/"),
    Item("arrow", "PyArrow", "zero-copy buffers", "py",
         "docs/interop/np-frombuffer.md#pyarrow", "https://arrow.apache.org/docs/python/"),
    Item("polars", "polars", "zero-copy Series", "py",
         "docs/interop/np-frombuffer.md#seven-libraries-verbatim", "https://pola.rs/"),
    Item("ort", "ONNX Runtime", "OrtValue · Run()", "net",
         "docs/interop/onnxruntime.md", "https://onnxruntime.ai/"),
    Item("mlnet", "ML.NET", "IDataView", "net",
         "docs/interop/mlnet.md", "https://dotnet.microsoft.com/apps/ai/ml-dotnet"),
    Item("tensors", "System.Numerics.Tensors", "Tensor<T> · TensorSpan<T>", "net",
         "docs/interop/system-numerics-tensors.md", "https://www.nuget.org/packages/System.Numerics.Tensors"),
    Item("torchsharp", "TorchSharp", "zero-copy torch.Tensor", "net",
         "docs/interop/torchsharp.md", "https://github.com/dotnet/TorchSharp"),
    Item("drawing", "System.Drawing", "Bitmap ⇄ NDArray", "net",
         "docs/extensions/bitmap.md", "https://learn.microsoft.com/dotnet/api/system.drawing"),
    Item("numpynet", "Numpy.NET", "PyObject handoff", "net",
         "docs/interop/numpy-net.md", "https://github.com/SciSharp/Numpy.NET"),
    Item("openblas", "OpenBLAS", "BLAS · LAPACK", "native",
         "docs/interop/openblas.md", "https://www.openblas.net/"),
    Item("npy", ".npy / .npz", "byte-exact np.save", "file",
         "docs/fundamentals/io.md", "https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html"),
]
BY_KEY = {item.key: item for item in ITEMS}

# NumSharp's own octagon on the home page: "Open Library" is the repository, "Open Docs" (and a click on the octagon) the
# Interoperability guide, which introduces every box around it.
CORE = Item("numsharp", "NumSharp", "NumPy for .NET", "core",
            "docs/interop/index.md", "https://github.com/SciSharp/NumSharp")

# Each side in reading order (top and bottom left to right, left and right top to bottom). Going round the ring
# clockwise, Python fills the bottom-left, the left column and the top-left; native, .NET and files follow on the right.
SIDES = {
    "top": ["pillow", "opencv", "openblas"],
    "right": ["ort", "mlnet", "tensors", "torchsharp", "drawing", "numpynet"],
    "bottom": ["polars", "arrow", "npy"],
    "left": ["numpy", "torch", "pandas", "scipy", "sklearn", "mpl"],
}

ECOSYSTEMS = {
    "py": dict(color="#FFD43B", legend="Python · NumSharp.Interop.pythonnet"),
    "net": dict(color="#3FC6EE", legend=".NET · NumSharp.Interop.* · NumSharp.Bitmap"),
    "native": dict(color="#3DFF9E", legend="Native · NumSharp.Interop.OpenBLAS"),
    "file": dict(color="#FF5FD7", legend="Files · NumSharp core"),
}
LEGEND_ROWS = [["py", "net"], ["native", "file"]]

# The figlet "future" letters, three rows each, in heavy box drawing.
FUTURE = {
    "N": ["┏┓╻", "┃┗┫", "╹ ╹"],
    "U": ["╻ ╻", "┃ ┃", "┗━┛"],
    "M": ["┏┳┓", "┃┃┃", "╹ ╹"],
    "S": ["┏━┓", "┗━┓", "┗━┛"],
    "H": ["╻ ╻", "┣━┫", "╹ ╹"],
    "A": ["┏━┓", "┣━┫", "╹ ╹"],
    "R": ["┏━┓", "┣┳┛", "╹┗╸"],
    "P": ["┏━┓", "┣━┛", "╹  "],
}

# Which half-strokes (Up, Down, Left, Right, from the cell center) make each heavy box-drawing character.
ARMS = {
    "━": "LR", "┃": "UD", "┏": "DR", "┓": "DL", "┗": "UR", "┛": "UL",
    "┣": "UDR", "┫": "UDL", "┳": "LRD", "┻": "LRU", "╋": "UDLR", "╹": "U", "╻": "D", "╸": "L", "╺": "R",
}

# ------------------------------------------------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """A rectangle of grid cells; an octagon is drawn inside it along the center lines of its outer cells.

    Attributes:
        col: The leftmost cell column.
        row: The top cell row.
        w: The width in cells.
        h: The height in cells.
    """
    col: int
    row: int
    w: int
    h: int

    @property
    def cx(self):
        """The horizontal center, in cells."""
        return self.col + self.w / 2

    @property
    def cy(self):
        """The vertical center, in cells."""
        return self.row + self.h / 2

    def overlaps(self, other):
        """Whether two boxes share a cell; touching edges do not count.

        Args:
            other: The box to test against.

        Returns:
            True when the two cell rectangles intersect.
        """
        return (self.col < other.col + other.w and other.col < self.col + self.w
                and self.row < other.row + other.h and other.row < self.row + self.h)


@dataclass(frozen=True)
class Layout:
    """Where everything goes.

    Attributes:
        rows: The grid's height in cells, legend included.
        center: NumSharp's octagon.
        sats: Each integration's octagon, by key.
        cables: Each integration's cable as the four control points of a cubic Bezier curve in px, from its port on
            NumSharp's octagon to its port on the integration's octagon.
    """
    rows: int
    center: Box
    sats: dict
    cables: dict


def satellite_width(item):
    """The width in cells of an integration's octagon: its content plus a border cell and a blank cell on each side.

    Args:
        item: The integration.

    Returns:
        The width that fits the icon and the large name on one line and the caption on the next.
    """
    content = max(ICON_PX + ICON_GAP + len(item.name) * NAME_ADV, len(item.caption) * CW)
    return math.ceil(content / CW) + 4


def make_layout():
    """Places NumSharp in the middle, the integrations on a ring around it, and a cable from one to each.

    Top and bottom octagons sit in one row each, centered, 34 cells apart; left and right octagons are spread evenly
    down the two columns. Every side of NumSharp's octagon gets one port per octagon on that side, spread evenly along
    the side in the same order, and each cable leaves its port at right angles and reaches its octagon at right angles,
    so the cables of one side fan out without crossing.

    Returns:
        The layout.

    Raises:
        ValueError: When two octagons overlap, an octagon leaves the canvas, or SIDES names an unknown key or misses one
            (a renamed or added item must be placed).
    """
    placed = [key for keys in SIDES.values() for key in keys]
    if sorted(placed) != sorted(BY_KEY):
        raise ValueError(f"SIDES must place every item exactly once: placed {sorted(placed)}, items {sorted(BY_KEY)}")

    center = Box((COLS - CENTER_COLS) // 2, (RING_ROWS - CENTER_ROWS) // 2, CENTER_COLS, CENTER_ROWS)
    sats = {}
    for side, row in (("top", 0), ("bottom", RING_ROWS - SAT_ROWS)):
        keys = SIDES[side]
        for i, key in enumerate(keys):
            w = satellite_width(BY_KEY[key])
            cx = COLS / 2 + (i - (len(keys) - 1) / 2) * 34
            sats[key] = Box(round(cx - w / 2), row, w, SAT_ROWS)
    first, last = 6, RING_ROWS - 6 - SAT_ROWS
    for side in ("left", "right"):
        keys = SIDES[side]
        for i, key in enumerate(keys):
            w = satellite_width(BY_KEY[key])
            # spread from the first to the last row; a column of one octagon sits halfway
            row = first + round(i * (last - first) / (len(keys) - 1)) if len(keys) > 1 else (first + last) // 2
            sats[key] = Box(0 if side == "left" else COLS - w, row, w, SAT_ROWS)

    boxes = list(sats.items()) + [("NumSharp", center)]
    for i, (a_key, a) in enumerate(boxes):
        if a.col < 0 or a.row < 0 or a.col + a.w > COLS or a.row + a.h > RING_ROWS:
            raise ValueError(f"the octagon of {a_key} leaves the {COLS} x {RING_ROWS} ring: {a}")
        for b_key, b in boxes[i + 1:]:
            if a.overlaps(b):
                raise ValueError(f"the octagons of {a_key} and {b_key} overlap: {a} and {b}")

    cables = {}
    for side, keys in SIDES.items():
        for i, key in enumerate(keys):
            cables[key] = _cable(center, sats[key], side, (i + 0.5) / len(keys))
    return Layout(RING_ROWS + 3, center, sats, cables)


def _cable(center, sat, side, along):
    """The Bezier control points of one cable.

    Args:
        center: NumSharp's octagon.
        sat: The integration's octagon.
        side: Which side of NumSharp the integration is on ('top', 'right', 'bottom' or 'left').
        along: Where the port sits along that side of NumSharp's octagon, 0..1 in reading order.

    Returns:
        Four (x, y) points in px. The handles run along the axis that leaves the side, 55 % of the distance along it,
        which gives an S-curve that meets both octagons square.
    """
    if side in ("top", "bottom"):
        lo, hi = center.col + 5, center.col + center.w - 5
        x0 = lo + (hi - lo) * along
        y0 = center.row + 0.5 if side == "top" else center.row + center.h - 0.5
        x3 = sat.cx
        y3 = sat.row + sat.h - 0.5 if side == "top" else sat.row + 0.5
    else:
        lo, hi = center.row + 1.5, center.row + center.h - 1.5
        y0 = lo + (hi - lo) * along
        x0 = center.col + 0.5 if side == "left" else center.col + center.w - 0.5
        y3 = sat.cy
        x3 = sat.col + sat.w - 0.5 if side == "left" else sat.col + 0.5
    p0, p3 = (x0 * CW, y0 * CH), (x3 * CW, y3 * CH)
    if side in ("top", "bottom"):
        k = (p3[1] - p0[1]) * 0.55
        return p0, (p0[0], p0[1] + k), (p3[0], p3[1] - k), p3
    k = (p3[0] - p0[0]) * 0.55
    return p0, (p0[0] + k, p0[1]), (p3[0] - k, p3[1]), p3


def curve_length(points):
    """The arc length of a cubic Bezier curve, from 400 chords (far below a pixel of error at this scale).

    Args:
        points: The four control points.

    Returns:
        The length in px.
    """
    def at(t):
        """The curve's point at parameter t (0 at the first control point, 1 at the last), Bernstein form."""
        u = 1 - t
        return tuple(u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d for a, b, c, d in zip(*points))
    samples = [at(i / 400) for i in range(401)]
    return sum(math.dist(samples[i - 1], samples[i]) for i in range(1, len(samples)))


# ------------------------------------------------------------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------------------------------------------------------------


def future_rows(word):
    """A word in the "future" letters.

    Args:
        word: Upper-case letters that FUTURE has.

    Returns:
        Three strings, one per row, with one blank column between letters.

    Raises:
        KeyError: When a letter is missing from FUTURE.
    """
    return [" ".join(FUTURE[ch][r] for ch in word) for r in range(3)]


def heavy_glyph(ch, x, y):
    """Path data that draws a heavy box-drawing character the way a font does: half-strokes from the cell center.

    Args:
        ch: A character from ARMS.
        x: The cell's left edge in px.
        y: The cell's top edge in px.

    Returns:
        SVG path data (move-to / line-to pairs).
    """
    mx, my = x + CW / 2, y + CH / 2
    ends = {"U": (mx, y), "D": (mx, y + CH), "L": (x, my), "R": (x + CW, my)}
    return "".join(f"M{mx:g},{my:g}L{ends[arm][0]:g},{ends[arm][1]:g}" for arm in ARMS[ch])


def octagon_points(box, inset=0.0):
    """The octagon inside a box: sides on the center lines of the outer cells, 45-degree chamfers one cell across.

    Args:
        box: The box.
        inset: How far inside that outline to draw, in px (NumSharp's octagon has a second, inner frame).

    Returns:
        The polygon's points attribute.
    """
    x0, x1 = (box.col + 0.5) * CW + inset, (box.col + box.w - 0.5) * CW - inset
    y0, y1 = (box.row + 0.5) * CH + inset, (box.row + box.h - 0.5) * CH - inset
    k = CW
    points = [(x0, y0 + k), (x0 + k, y0), (x1 - k, y0), (x1, y0 + k),
              (x1, y1 - k), (x1 - k, y1), (x0 + k, y1), (x0, y1 - k)]
    return " ".join(f"{px:g},{py:g}" for px, py in points)


def text_run(text, x, y, advance, css_class, fill):
    """A line of text with every character at its own x, so it keeps its width in any font.

    Each character is centered in a slot `advance` px wide (text-anchor="middle" on a list of x positions): a font
    whose glyphs are narrower or wider than the slot leaves the next character where it belongs, so the line never
    runs into an octagon's border, whatever font the reader's system substitutes.

    Args:
        text: The characters.
        x: The left edge of the first character's slot, in px.
        y: The baseline, in px.
        advance: The width of one character's slot, in px.
        css_class: The class that sets the font.
        fill: The color.

    Returns:
        A <text> element.
    """
    xs = " ".join(f"{x + advance * (i + 0.5):g}" for i in range(len(text)))
    return (f'<text class="{css_class}" x="{xs}" y="{y:g}" text-anchor="middle" fill="{fill}">'
            f"{html.escape(text)}</text>")


def python_icon(x, y, s):
    """The two-snake Python mark in its own blue and yellow.

    Args:
        x: Left edge in px.
        y: Top edge in px.
        s: Size in px (the mark is square).

    Returns:
        An SVG group.
    """
    blue = ("M54.9,0C50.3,0,46,0.4,42.1,1.1C30.8,3.1,28.7,7.3,28.7,15V25.3H55.5V28.7H28.7H18.6"
            "C10.8,28.7,4,33.4,1.9,42.3C-0.6,52.5-0.7,58.9,1.9,69.5C3.8,77.5,8.4,83.1,16.2,83.1H25.4V70.9"
            "C25.4,62,33,54.2,42.1,54.2H68.9C76.4,54.2,82.3,48.1,82.3,40.6V15.1C82.3,7.8,76.2,2.3,68.9,1.1"
            "C64.3,0.3,59.5,0,54.9,0ZM40.4,8.2C43.2,8.2,45.5,10.5,45.5,13.3C45.5,16.2,43.2,18.4,40.4,18.4"
            "C37.7,18.4,35.4,16.2,35.4,13.3C35.4,10.5,37.7,8.2,40.4,8.2Z")
    yellow = ("M85.6,28.7V40.6C85.6,49.8,77.8,57.6,68.9,57.6H42.1C34.8,57.6,28.7,63.8,28.7,71.2V96.7"
              "C28.7,104,35,108.3,42.1,110.3C50.6,112.8,58.7,113.3,68.9,110.3C75.6,108.4,82.3,104.4,82.3,96.7"
              "V86.5H55.5V83.1H82.3H95.7C103.5,83.1,106.4,77.7,109.1,69.5C111.9,61.1,111.8,53,109.1,42.3"
              "C107.2,34.5,103.5,28.7,95.7,28.7ZM70.6,93.3C73.3,93.3,75.6,95.6,75.6,98.4C75.6,101.2,73.3,103.5,70.6,103.5"
              "C67.8,103.5,65.5,101.2,65.5,98.4C65.5,95.6,67.8,93.3,70.6,93.3Z")
    return (f'<g transform="translate({x:g},{y:g}) scale({s / 112:.4f})">'
            f'<path d="{blue}" fill="#4B8BEF"/><path d="{yellow}" fill="#FFD43B"/></g>')


def net_icon(x, y, s):
    """The .NET badge: white ".NET" on .NET's purple.

    Args:
        x: Left edge in px.
        y: Top edge in px.
        s: Size in px.

    Returns:
        An SVG group.
    """
    return (f'<g transform="translate({x:g},{y:g})"><rect width="{s}" height="{s}" rx="{s * 0.2:.1f}" fill="#512BD4"/>'
            f'<text x="{s / 2:g}" y="{s * 0.64:.1f}" text-anchor="middle" '
            f'font-family="Segoe UI,Helvetica,Arial,sans-serif" font-weight="700" font-size="{s * 0.37:.1f}" '
            f'fill="#fff">.NET</text></g>')


def chip_icon(x, y, s):
    """A processor chip, for the native BLAS.

    Args:
        x: Left edge in px.
        y: Top edge in px.
        s: Size in px.

    Returns:
        An SVG group.
    """
    m = s * 0.24
    color = ECOSYSTEMS["native"]["color"]
    parts = [f'<rect x="{m:.1f}" y="{m:.1f}" width="{s - 2 * m:.1f}" height="{s - 2 * m:.1f}" rx="1.5" '
             f'fill="#0b2a1c" stroke="{color}" stroke-width="1.5"/>',
             f'<rect x="{s * 0.41:.1f}" y="{s * 0.41:.1f}" width="{s * 0.18:.1f}" height="{s * 0.18:.1f}" fill="{color}"/>']
    for i in range(3):
        o = m + (s - 2 * m) * (i + 1) / 4
        parts.append(f'<path d="M{o:.1f},1V{m:.1f}M{o:.1f},{s - m:.1f}V{s - 1}M1,{o:.1f}H{m:.1f}M{s - m:.1f},{o:.1f}'
                     f'H{s - 1}" stroke="{color}" stroke-width="1.2"/>')
    return f'<g transform="translate({x:g},{y:g})">{"".join(parts)}</g>'


def file_icon(x, y, s):
    """A page with a folded corner, for the file formats.

    Args:
        x: Left edge in px.
        y: Top edge in px.
        s: Size in px.

    Returns:
        An SVG group.
    """
    f = s * 0.3
    color = ECOSYSTEMS["file"]["color"]
    return (f'<g transform="translate({x:g},{y:g})" fill="none" stroke="{color}" stroke-width="1.5" '
            f'stroke-linejoin="round">'
            f'<path d="M{s * 0.2:.1f},1H{s * 0.62:.1f}L{s * 0.82:.1f},{f:.1f}V{s - 1}H{s * 0.2:.1f}Z" fill="#2a0b24"/>'
            f'<path d="M{s * 0.62:.1f},1V{f:.1f}H{s * 0.82:.1f}"/>'
            f'<path d="M{s * 0.32:.1f},{s * 0.55:.1f}H{s * 0.7:.1f}M{s * 0.32:.1f},{s * 0.72:.1f}H{s * 0.7:.1f}" '
            f'stroke-width="1.2"/></g>')


ICONS = {"py": python_icon, "net": net_icon, "native": chip_icon, "file": file_icon}


def numsharp_logo(x, y, s):
    """NumSharp's logo, read from the repository's icon so the picture follows any change to it.

    The icon is an Inkscape file: its drawing layer is kept as it is (paths, colors, transforms) and everything
    Inkscape adds for itself (metadata, named views, inkscape: and sodipodi: attributes) is dropped.

    Args:
        x: Left edge in px.
        y: Top edge in px.
        s: Size in px (the icon is square).

    Returns:
        A nested <svg> element that draws the logo at that place and size.

    Raises:
        OSError: When the icon file cannot be read.
        ValueError: When the icon has no viewBox or no drawing layer (it is no longer the Inkscape export this reads).
    """
    svg_ns = "http://www.w3.org/2000/svg"
    root = ET.parse(LOGO).getroot()
    view_box = root.get("viewBox")
    layers = [g for g in root.iter(f"{{{svg_ns}}}g") if g.get("id") == "layer1"]
    if not view_box or not layers:
        raise ValueError(f"{LOGO} has no viewBox or no drawing layer 'layer1'")

    def clean(element):
        """Strips one element and its subtree, in place, down to what draws.

        Args:
            element: The element; its namespaced attributes and its non-SVG children (Inkscape's and Sodipodi's
                bookkeeping) are removed, its SVG children are cleaned the same way.
        """
        # An attribute in a namespace is Inkscape's or Sodipodi's bookkeeping; the drawing needs none of them.
        for name in [n for n in element.attrib if n.startswith("{")]:
            del element.attrib[name]
        for child in list(element):
            if not child.tag.startswith(f"{{{svg_ns}}}"):
                element.remove(child)
            else:
                clean(child)

    layer = layers[0]
    clean(layer)
    body = ET.tostring(layer, encoding="unicode")
    # ElementTree writes the SVG namespace as a prefix; inside our SVG the default namespace already is SVG.
    body = re.sub(r"\s?xmlns:ns0=\"[^\"]*\"", "", body).replace("ns0:", "")
    return f'<svg x="{x:g}" y="{y:g}" width="{s:g}" height="{s:g}" viewBox="{view_box}">{body}</svg>'


def render(layout):
    """The whole picture as an SVG document.

    Args:
        layout: Where everything goes (make_layout()).

    Returns:
        The SVG text, deterministic for one input: running the script twice writes the same bytes.

    Raises:
        OSError: When the NumSharp logo cannot be read (numsharp_logo()).
        ValueError: When the logo file is not the Inkscape export numsharp_logo() reads.
    """
    width, height = COLS * CW + 2 * PAD, layout.rows * CH + 2 * PAD
    names = ", ".join(item.name for item in ITEMS)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-labelledby="title desc">',
        "<title id=\"title\">NumSharp integrations</title>",
        f'<desc id="desc">NumSharp in the center, with a data stream out to each integration: {html.escape(names)}.</desc>',
        "<!-- Generated by tools/gen_integrations_diagram.py; edit the script, not this file. -->",
        "<defs>",
        '<radialGradient id="bg" cx="50%" cy="47%" r="72%"><stop offset="0" stop-color="#1d1145"/>'
        '<stop offset=".55" stop-color="#0b0720"/><stop offset="1" stop-color="#05030d"/></radialGradient>',
        f'<pattern id="cells" width="{CW}" height="{CH}" patternUnits="userSpaceOnUse" x="{PAD}" y="{PAD}">'
        f'<circle cx="{CW / 2}" cy="{CH / 2}" r="0.65" fill="#2b1f5c"/></pattern>',
        # glow: the shape, plus two blurred copies of itself under it
        '<filter id="glow" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="2.4" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<filter id="soft" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="1.6" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<filter id="halo" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="5" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        "<style>"
        f".name{{font-family:{FONT};font-size:{NAME_PX}px;font-weight:700}}"
        f".cap{{font-family:{FONT};font-size:{CAPTION_PX}px}}"
        f".tag{{font-family:{FONT};font-size:15.5px;font-weight:700;letter-spacing:.5px}}"
        # one burst per cable every 3 s, all cables in step, so the bursts leave NumSharp together
        "@keyframes burst{from{stroke-dashoffset:0}to{stroke-dashoffset:-100}}"
        ".burst{animation:burst 3s linear infinite}"
        "@media (prefers-reduced-motion:reduce){.burst{animation:none;visibility:hidden}}"
        "</style>",
    ]

    # cable colors: NumSharp's purple at the port, the integration's color from 70 % of the way out
    gradients = []
    for key, (p0, _, _, p3) in layout.cables.items():
        color = ECOSYSTEMS[BY_KEY[key].eco]["color"]
        gradients.append(f'<linearGradient id="c-{key}" gradientUnits="userSpaceOnUse" x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" '
                         f'x2="{p3[0]:.1f}" y2="{p3[1]:.1f}"><stop offset="0" stop-color="{NUMSHARP}"/>'
                         f'<stop offset=".7" stop-color="{color}"/></linearGradient>')
    out.append("".join(gradients))
    out.append("</defs>")
    out.append(f'<rect width="{width}" height="{height}" rx="16" fill="url(#bg)"/>')
    out.append(f'<rect width="{width}" height="{height}" rx="16" fill="url(#cells)"/>')
    out.append(f'<g transform="translate({PAD},{PAD})">')

    # cables: the dotted track (zero-length dashes with round caps are dots, one every 8.5 px), the burst (five bright
    # dots 8.5 px apart, then a gap longer than any cable, in units of pathLength="100" so one keyframe set serves every
    # cable), and a larger dot where the cable meets its octagon
    cables = []
    for key, points in layout.cables.items():
        p0, p1, p2, p3 = points
        d = f"M{p0[0]:.1f},{p0[1]:.1f}C{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}"
        step = 8.5 / curve_length(points) * 100
        burst = " ".join([f"0 {step:.3f}"] * 4) + " 0 200"
        color = ECOSYSTEMS[BY_KEY[key].eco]["color"]
        cables.append(f'<path d="{d}" fill="none" stroke="url(#c-{key})" stroke-width="3.2" stroke-linecap="round" '
                      f'stroke-dasharray="0 8.5"/>')
        cables.append(f'<path class="burst" d="{d}" pathLength="100" fill="none" stroke="#ffffff" stroke-width="4.2" '
                      f'stroke-linecap="round" stroke-dasharray="{burst}"/>')
        cables.append(f'<circle cx="{p3[0]:.1f}" cy="{p3[1]:.1f}" r="3" fill="{color}"/>')
    out.append(f'<g filter="url(#soft)">{"".join(cables)}</g>')

    # octagons: NumSharp's with a second, inner frame; each integration's in its ecosystem's color
    center = layout.center
    octagons = [f'<polygon points="{octagon_points(center)}" fill="#120a26" fill-opacity=".92" stroke="{NUMSHARP}" '
                f'stroke-width="3"/>',
                f'<polygon points="{octagon_points(center, 5)}" fill="none" stroke="{NUMSHARP}" stroke-opacity=".45" '
                f'stroke-width="1"/>']
    for key, box in layout.sats.items():
        color = ECOSYSTEMS[BY_KEY[key].eco]["color"]
        octagons.append(f'<polygon points="{octagon_points(box)}" fill="#07051a" fill-opacity=".92" stroke="{color}" '
                        f'stroke-width="1.6"/>')
    out.append(f'<g filter="url(#glow)">{"".join(octagons)}</g>')

    # NumSharp: the logo, the NUMSHARP wordmark in "future" letters, the tagline under the wordmark. The wordmark starts
    # 13 cells in (2.6 cells of margin, the 9.2-cell logo, a cell of air) and spans rows 3-5 of the 11; the tagline sits
    # on row 7, so logo, wordmark and tagline share the octagon's vertical center.
    title = future_rows("NUMSHARP")
    title_col = center.col + 13
    title_row = center.row + 3
    glyphs = []
    for r, line in enumerate(title):
        for c, ch in enumerate(line):
            if ch in ARMS:
                glyphs.append(heavy_glyph(ch, (title_col + c) * CW, (title_row + r) * CH))
    out.append(f'<g filter="url(#glow)"><path d="{"".join(glyphs)}" fill="none" stroke="{NUMSHARP_TEXT}" '
               f'stroke-width="3.2" stroke-linecap="square"/></g>')
    tagline = "NumPy for .NET"
    title_mid = (title_col + len(title[0]) / 2) * CW
    tagline_x = title_mid - len(tagline) * CW / 2
    out.append(f'<g filter="url(#soft)">{text_run(tagline, tagline_x, (center.row + 7) * CH + 15, CW, "tag", "#F1E9FF")}</g>')
    logo_x = (center.col + 2.6) * CW
    logo_y = center.cy * CH - LOGO_PX / 2
    out.append(f'<g filter="url(#halo)">{numsharp_logo(logo_x, logo_y, LOGO_PX)}</g>')

    # each integration: icon and name on the first line, caption on the second
    labels, icons = [], []
    for key, box in layout.sats.items():
        item = BY_KEY[key]
        x = (box.col + 2) * CW
        top = box.row * CH
        icons.append(ICONS[item.eco](x, top + 19, ICON_PX))
        labels.append(text_run(item.name, x + ICON_PX + ICON_GAP, top + 38, NAME_ADV, "name", "#F7F5FF"))
        labels.append(text_run(item.caption, x, top + 60, CW, "cap", ECOSYSTEMS[item.eco]["color"]))

    # legend: two rows of two entries under the ring, the block centered
    first_col = max(len(ECOSYSTEMS[row[0]]["legend"]) for row in LEGEND_ROWS) + 3 + 6
    block = first_col + max(len(ECOSYSTEMS[row[1]]["legend"]) for row in LEGEND_ROWS) + 3
    left = (COLS - block) / 2
    for r, row in enumerate(LEGEND_ROWS):
        for c, eco in enumerate(row):
            x = (left + c * first_col) * CW
            y = (RING_ROWS + 1 + r) * CH
            icons.append(ICONS[eco](x, y + 2, 16))
            labels.append(text_run(ECOSYSTEMS[eco]["legend"], x + 24, y + 15, CW, "cap", ECOSYSTEMS[eco]["color"]))
    out.append(f'<g filter="url(#soft)">{"".join(labels)}</g>')
    out.append("".join(icons))
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def render_links(layout):
    """The home page's link layer over the picture: one hotspot per octagon, NumSharp's included, as HTML.

    Each hotspot is a box positioned in percentages of the picture, so it scales with the <img> under it at any width.
    It holds a link that covers the octagon (a click opens the docs page; it stays out of the tab order and away from
    screen readers because "Open Docs" repeats it) with an SVG outline of the same octagon, which the stylesheet rings
    on hover, and the two buttons "Open Library" and "Open Docs", which the stylesheet shows under the octagon on hover
    or when a button has the keyboard focus. Each button's visually hidden suffix names the library, so a screen reader
    hears "Open Docs PyTorch" instead of eighteen identical "Open Docs".

    The side class tells the stylesheet where the buttons fit: under the left column they start at the octagon's left
    edge and under the right column they end at its right edge, so they never leave the picture; elsewhere they are
    centered.

    Args:
        layout: Where everything goes (make_layout()); the same layout render() draws, so the boxes match the octagons.

    Returns:
        The HTML lines between the markers, without the markers, ending in a newline; the same bytes for one input.
    """
    width, height = COLS * CW + 2 * PAD, layout.rows * CH + 2 * PAD
    sides = {key: side for side, keys in SIDES.items() for key in keys}
    entries = [(CORE, layout.center, "center")] + [(BY_KEY[key], box, sides[key]) for key, box in layout.sats.items()]

    def pct(value, whole):
        """A CSS percentage of the picture, to four decimals without trailing zeros.

        Four decimals place a box within 0.06 px on the 1324 px picture; trimming the zeros keeps the generated lines
        short and the output byte-stable (17.145% stays 17.145%, not 17.1450%).

        Args:
            value: A length or position in the picture's px.
            whole: The picture's width or height in px.

        Returns:
            The value as a percentage string such as "17.145%".
        """
        return f"{value / whole * 100:.4f}".rstrip("0").rstrip(".") + "%"

    lines = ["<!-- Generated by tools/gen_integrations_diagram.py from its ITEMS and layout; edit the script, then run it. -->",
             '<div class="ns-hot-layer">']
    for item, box, side in entries:
        # The octagon's outline exactly as octagon_points() draws it, moved by the card's margin and expressed in the
        # hotspot's own coordinates (its top-left corner is the octagon's left and top sides).
        x0, x1 = PAD + (box.col + 0.5) * CW, PAD + (box.col + box.w - 0.5) * CW
        y0, y1 = PAD + (box.row + 0.5) * CH, PAD + (box.row + box.h - 0.5) * CH
        w, h, k = x1 - x0, y1 - y0, CW
        points = [(0, k), (k, 0), (w - k, 0), (w, k), (w, h - k), (w - k, h), (k, h), (0, h - k)]
        outline = " ".join(f"{px:g},{py:g}" for px, py in points)
        style = f"left:{pct(x0, width)};top:{pct(y0, height)};width:{pct(w, width)};height:{pct(h, height)}"
        name = html.escape(item.name)
        docs, site = html.escape(item.docs), html.escape(item.site)
        lines.append(
            f'<div class="ns-hot ns-hot-{item.eco} ns-hot-{side}" style="{style}">'
            f'<a class="ns-hot-box" href="{docs}" tabindex="-1" aria-hidden="true">'
            f'<svg viewBox="0 0 {w:g} {h:g}" preserveAspectRatio="none" focusable="false"><polygon points="{outline}"/></svg></a>'
            f'<span class="ns-hot-menu">'
            f'<a class="ns-hot-btn" href="{site}"><span>Open Library</span><span class="visually-hidden"> {name}</span></a>'
            f'<a class="ns-hot-btn" href="{docs}"><span>Open Docs</span><span class="visually-hidden"> {name}</span></a>'
            f'</span></div>')
    lines.append("</div>")
    return "\n".join(lines) + "\n"


def splice_home(text, block):
    """The home page with its link layer replaced by a fresh one.

    Args:
        text: The home page's current Markdown (docs/website-src/index.md).
        block: The new layer (render_links()), which goes between the two marker lines.

    Returns:
        The home page with everything between the markers replaced; the rest of the file is untouched.

    Raises:
        ValueError: When a marker is missing or appears more than once, or the end marker comes first: rewriting a
            guess would destroy hand-written content, so the script stops instead.
    """
    begin, end = text.count(INTEGRATIONS_LINKS_BEGIN), text.count(INTEGRATIONS_LINKS_END)
    if begin != 1 or end != 1:
        raise ValueError(f"{os.path.relpath(HOME, ROOT)} must hold {INTEGRATIONS_LINKS_BEGIN} and "
                         f"{INTEGRATIONS_LINKS_END} exactly once each (found {begin} and {end})")
    head, rest = text.split(INTEGRATIONS_LINKS_BEGIN)
    old, tail = rest.split(INTEGRATIONS_LINKS_END)
    if not old.startswith("\n"):
        raise ValueError(f"{INTEGRATIONS_LINKS_BEGIN} must end its line in {os.path.relpath(HOME, ROOT)}")
    return f"{head}{INTEGRATIONS_LINKS_BEGIN}\n{block}{INTEGRATIONS_LINKS_END}{tail}"


def main(argv):
    """Writes the SVG and the home page's link layer, or with --check compares the committed ones to a fresh render.

    Args:
        argv: The command-line arguments after the script name.

    Returns:
        The process exit code: 0 on success, 1 when --check finds the committed SVG or link layer out of date.

    Raises:
        ValueError: From make_layout() or numsharp_logo() when the layout or the logo is unusable, or from splice_home()
            when the home page's markers are missing or repeated.
        OSError: When the logo or the home page cannot be read, or a file cannot be written.
    """
    layout = make_layout()
    svg = render(layout)
    with open(HOME, encoding="utf-8", newline="") as f:
        home = f.read()
    # newline="" keeps the home page's own line endings: a CRLF checkout must compare and write back byte-identical
    # outside the generated block, and the block itself uses the file's ending so the file never mixes the two.
    eol = "\r\n" if "\r\n" in home else "\n"
    new_home = splice_home(home, render_links(layout).replace("\n", eol))
    if "--check" in argv:
        try:
            with open(OUT, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = None
        stale = [path for path, fresh, now in ((OUT, svg, current), (HOME, new_home, home)) if fresh != now]
        for path in stale:
            print(f"{os.path.relpath(path, ROOT)} is out of date: run python tools/gen_integrations_diagram.py")
        if stale:
            return 1
        print(f"{os.path.relpath(OUT, ROOT)} and the link layer of {os.path.relpath(HOME, ROOT)} are up to date")
        return 0
    # newline="\n": the same bytes on every OS, so --check holds across machines
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(svg)
    with open(HOME, "w", encoding="utf-8", newline="") as f:
        f.write(new_home)
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({len(svg.encode('utf-8')):,} bytes, {len(ITEMS)} integrations) and the "
          f"link layer of {os.path.relpath(HOME, ROOT)} ({len(ITEMS) + 1} hotspots)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
