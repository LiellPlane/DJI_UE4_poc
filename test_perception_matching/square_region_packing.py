from __future__ import annotations
from dataclasses import dataclass, field
import heapq
import itertools
from typing import List


# ────────────────────────────────
# Geometry dataclasses
# ────────────────────────────────
@dataclass(frozen=True, slots=True)
class Canvas:
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Region:
    """
    Axis‑aligned rectangle (also used for reserved areas).
    Coordinates are top‑left (x, y); size is width × height.
    """
    x: int
    y: int
    width: int
    height: int
    _priority: int = field(init=False, repr=False)

    def __post_init__(self):
        # Negative so that 'largest side' becomes lowest numeric value in the heap
        object.__setattr__(self, "_priority", -min(self.width, self.height))

    # ── geometry helpers ──────────────────────────────────────────────
    def intersect(self, other: "Region") -> "Region | None":
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2 = min(self.x + self.width, other.x + other.width)
        iy2 = min(self.y + self.height, other.y + other.height)
        if ix1 >= ix2 or iy1 >= iy2:
            return None
        return Region(ix1, iy1, ix2 - ix1, iy2 - iy1)

    def subtract(self, cutter: "Region") -> List["Region"]:
        """Return the list of parts of *self* that are not removed by *cutter*."""
        inter = self.intersect(cutter)
        if not inter:                       # no overlap
            return [self]

        pieces: List[Region] = []
        # Horizontal slices
        if self.y < inter.y:  # top
            pieces.append(Region(self.x, self.y, self.width, inter.y - self.y))
        if inter.y + inter.height < self.y + self.height:  # bottom
            pieces.append(
                Region(
                    self.x,
                    inter.y + inter.height,
                    self.width,
                    self.y + self.height - (inter.y + inter.height),
                )
            )
        # Vertical slices between top & bottom
        if self.x < inter.x:  # left
            pieces.append(
                Region(self.x, inter.y, inter.x - self.x, inter.height)
            )
        if inter.x + inter.width < self.x + self.width:  # right
            pieces.append(
                Region(
                    inter.x + inter.width,
                    inter.y,
                    self.x + self.width - (inter.x + inter.width),
                    inter.height,
                )
            )
        return pieces


@dataclass(frozen=True, slots=True)
class Square:
    x: int
    y: int
    side: int

    def __hash__(self):
        return hash((self.x, self.y, self.side))

    def __eq__(self, other):
        if not isinstance(other, Square):
            return NotImplemented
        return (self.x, self.y, self.side) == (other.x, other.y, other.side)


# ────────────────────────────────
# Packing algorithm
# ────────────────────────────────
def pack_squares(
    canvas: Canvas,
    reserved: List[Region],
    max_side_ratio: float = 0.5,
) -> List[Square]:
    """
    Fill the canvas with non‑overlapping maximal squares.

    Parameters
    ----------
    canvas : Canvas
        The drawing surface.
    reserved : list[Region]
        Already‑occupied areas to exclude.
    max_side_ratio : float, optional
        Upper bound for a square's side as a fraction of
        min(canvas.width, canvas.height).  Default = 0.5 (50 %).

    Returns
    -------
    list[Square]
    """
    # 1. Remove reserved areas from the canvas, producing free rectangles
    free: List[Region] = [Region(0, 0, canvas.width, canvas.height)]
    for r in reserved:
        new_free: List[Region] = []
        for rect in free:
            new_free.extend(rect.subtract(r))
        free = new_free

    # 2. Best‑first placement via a heap with a unique tie‑breaker
    max_side = int(min(canvas.width, canvas.height) * max_side_ratio)
    counter = itertools.count()  # unique increasing numbers
    heap: List[tuple[int, int, Region]] = [
        (region._priority, next(counter), region) for region in free
    ]
    heapq.heapify(heap)

    squares: List[Square] = []
    while heap:
        _, _, rect = heapq.heappop(heap)
        side = min(rect.width, rect.height, max_side)
        if side == 0:
            continue

        squares.append(Square(rect.x, rect.y, side))

        # Subtract the placed square and push leftovers back on the heap
        cutter = Region(rect.x, rect.y, side, side)
        for leftover in rect.subtract(cutter):
            if min(leftover.width, leftover.height) > 0:
                heapq.heappush(
                    heap, (leftover._priority, next(counter), leftover)
                )

    return squares


# # ────────────────────────────────
# # Demo / self‑test
# # ────────────────────────────────
# if __name__ == "__main__":
#     canvas = Canvas(800, 600)
#     reserved = [
#         Region(100, 100, 200, 150),
#         Region(500, 50, 180, 200),
#         Region(300, 400, 250, 120),
#     ]
#     for sq in pack_squares(canvas, reserved):
#         print(sq)



# ────────────────────────────────
# Optional visualisation (OpenCV)
# ────────────────────────────────
try:
    import cv2
    import numpy as np
except ImportError:   # keep the whole script usable even without OpenCV installed
    cv2 = None
    np  = None


def _hsv_cycle(n: int) -> list[tuple[int, int, int]]:
    """
    Return *n* distinct BGR colours produced from an HSV wheel,
    already converted for OpenCV (B, G, R in [0, 255]).
    """
    colours = []
    for i in range(n):
        hue = int(180 * i / n)          # OpenCV hue range: 0–179
        sat = 180                       # fairly vivid but not neon
        val = 230
        bgr = cv2.cvtColor(
            np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        colours.append(tuple(int(c) for c in bgr))   # convert to Python ints
    return colours


def visualise_packing(
    canvas: Canvas,
    reserved: list[Region],
    squares: list[Square],
    scale: int = 1,
    show: bool = True,
    save_path: str | None = None,
) -> np.ndarray | None:
    """
    Draw the canvas, reserved regions (gray), and square tiles (cycling colours).

    Parameters
    ----------
    canvas, reserved, squares : as produced by the packing algorithm.
    scale : int, optional
        Multiply all pixel dimensions by this factor for a larger display.
    show : bool, optional
        If True, call cv2.imshow / waitKey and destroyWindow.
    save_path : str or None, optional
        If given, save the visualisation to this path (PNG).

    Returns
    -------
    The resulting BGR image (numpy array) or None if cv2 is unavailable.
    """
    if cv2 is None:
        print("OpenCV not available; skipping visualisation.")
        return None

    W, H = canvas.width * scale, canvas.height * scale
    img = np.full((H, W, 3), 255, dtype=np.uint8)     # white background

    # Draw reserved regions first (filled, light gray)
    for r in reserved:
        cv2.rectangle(
            img,
            (r.x * scale, r.y * scale),
            ((r.x + r.width) * scale, (r.y + r.height) * scale),
            color=(180, 180, 180),
            thickness=-1,
        )

    # Draw square tiles (outline only) with a colour cycle
    palette = _hsv_cycle(max(8, len(squares)))  # at least 8 colours
    alpha = 0.25  # 25 % opaque fill
    for idx, sq in enumerate(squares):
        colour = palette[idx % len(palette)]
        top_left  = (sq.x * scale, sq.y * scale)
        bottom_rt = ((sq.x + sq.side) * scale, (sq.y + sq.side) * scale)

        # ---- NEW: translucent fill ----
        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_rt, colour, thickness=-1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

        # outline (kept as before)
        cv2.rectangle(img, top_left, bottom_rt, colour, thickness=max(2, scale))

    if save_path:
        cv2.imwrite(save_path, img)

    if show:
        cv2.imshow("Square Packing", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# ────────────────────────────────
# Quick test if run directly
# ────────────────────────────────
if __name__ == "__main__":
    # previous demo has already run; produce its data again
    canvas = Canvas(800, 600)
    reserved = [
        Region(100, 100, 200, 150),
        Region(500, 50, 180, 200),
        Region(300, 400, 250, 120),
    ]
    squares = pack_squares(canvas, reserved)

    # Visualise at 1 px : 1 unit (or change scale = 2 for larger view)
    visualise_packing(canvas, reserved, squares, scale=1, show=True)
