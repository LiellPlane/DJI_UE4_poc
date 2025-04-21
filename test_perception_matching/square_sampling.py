# ────────────────────────────────
# Sampling helper for “no‑a‑priori” matching
# ────────────────────────────────
import math
import random
from square_region_packing import Canvas, Region, Square, _hsv_cycle
import cv2
import numpy as np

def sample_square_regions(
    canvas: Canvas,
    *,
    max_side_ratio: float = 0.5,
    n_sizes: int = 10,
    positions_per_size: int = 20,
    rng_seed: int | None = None,
) -> list[Square]:
    """
    Return a *representative* but non‑exhaustive set of square regions.

    Parameters
    ----------
    canvas : Canvas
        Image extents.
    max_side_ratio : float, optional
        Largest side allowed as a fraction of min(canvas.width, canvas.height).
    n_sizes : int, optional
        How many distinct square side lengths to sample
        (geometric progression from large to small).
    positions_per_size : int, optional
        How many positions to keep for each size (picked at random from a grid).
    rng_seed : int or None, optional
        Seed for `random` so results can be reproduced.

    Returns
    -------
    list[Square]
    """
    rng = random.Random(rng_seed)

    max_side = int(min(canvas.width, canvas.height) * max_side_ratio)
    if max_side < 1:
        return []

    # Geometric progression of sizes (e.g. 256,128,64,32…)
    sizes = [max_side]
    ratio = 0.5 ** (1 / max(1, n_sizes - 1))
    while len(sizes) < n_sizes and sizes[-1] > 1:
        next_side = max(1, int(sizes[-1] * ratio))
        if next_side == sizes[-1]:
            break
        sizes.append(next_side)

    squares: list[Square] = []
    for side in sizes:
        # Grid spacing equal to side ⇒ non‑overlapping grid cells
        nx = math.ceil(canvas.width / side)
        ny = math.ceil(canvas.height / side)
        cells = [(ix, iy) for ix in range(nx) for iy in range(ny)]
        rng.shuffle(cells)
        for ix, iy in cells[:positions_per_size]:
            # Top‑left inside canvas
            x = min(ix * side, canvas.width  - side)
            y = min(iy * side, canvas.height - side)
            squares.append(Square(x, y, side))

    return squares


# ────────────────────────────────
# Visualiser for the sampled regions
# ────────────────────────────────
def visualise_sample(
    canvas: Canvas,
    squares: list[Square],
    *,
    scale: int = 1,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Draw only the sampled squares (coloured outlines).

    Parameters
    ----------
    canvas, squares : as returned by `sample_square_regions`.
    scale : int, optional
        Pixel multiplier for a larger view.
    show : bool, optional
        Call `cv2.imshow` / `waitKey`.
    save_path : str or None, optional
        Save the visualisation to a PNG if given.
    """
    if cv2 is None:
        print("OpenCV not available; skipping visualisation.")
        return None

    W, H = canvas.width * scale, canvas.height * scale
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    palette = _hsv_cycle(max(8, len(squares)))
    for idx, sq in enumerate(squares):
        colour = palette[idx % len(palette)]
        top_left  = (sq.x * scale, sq.y * scale)
        bottom_rt = ((sq.x + sq.side) * scale, (sq.y + sq.side) * scale)
        cv2.rectangle(img, top_left, bottom_rt, colour, max(2, scale))

    if save_path:
        cv2.imwrite(save_path, img)

    if show:
        cv2.imshow("Sampled Squares", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# ────────────────────────────────
# Extra demo (kept separate from earlier one)
# ────────────────────────────────
if __name__ == "__main__":
    print("\n--- Sampling demo ----------------------------------------")
    canvas = Canvas(800, 600)
    sampled = sample_square_regions(
        canvas,
        max_side_ratio=0.5,
        n_sizes=10,
        positions_per_size=20,
        rng_seed=42,
    )
    print(f"Generated {len(sampled)} sample regions")
    visualise_sample(canvas, sampled, scale=1, show=True)
