# ────────────────────────────────
# Sampling helper for "no‑a‑priori" matching
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
    min_side_ratio: float = 0.1,
    n_sizes: int = 10,
    min_distance_ratio: float = 0.1,  # Minimum distance between square centers as ratio of square size
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
    min_side_ratio : float, optional
        Smallest side allowed as a fraction of min(canvas.width, canvas.height).
    n_sizes : int, optional
        How many distinct square side lengths to sample
        (geometric progression from large to small).
    min_distance_ratio : float, optional
        Minimum distance between square centers as a ratio of the square size.
        For example, 0.1 means squares must be at least 10% of their size apart.
    rng_seed : int or None, optional
        Seed for `random` so results can be reproduced.

    Returns
    -------
    list[Square]
    """
    rng = random.Random(rng_seed)

    min_side = int(min(canvas.width, canvas.height) * min_side_ratio)
    max_side = int(min(canvas.width, canvas.height) * max_side_ratio)
    
    if max_side < 1 or min_side > max_side:
        return []

    # Geometric progression of sizes from max_side to min_side
    sizes = [max_side]
    ratio = (min_side / max_side) ** (1 / max(1, n_sizes - 1))
    while len(sizes) < n_sizes and sizes[-1] > min_side:
        next_side = max(min_side, int(sizes[-1] * ratio))
        if next_side == sizes[-1]:
            break
        sizes.append(next_side)

    squares: list[Square] = []
    for side in sizes:
        # Calculate minimum distance between square centers
        min_dist = side * min_distance_ratio
        
        # Create a grid with spacing based on minimum distance
        nx = math.ceil(canvas.width / min_dist)
        ny = math.ceil(canvas.height / min_dist)
        
        # Generate all possible positions
        cells = [(ix, iy) for ix in range(nx) for iy in range(ny)]
        rng.shuffle(cells)
        
        # Try to place squares while respecting minimum distance
        placed_squares = []
        for ix, iy in cells:
            x = min(ix * min_dist, canvas.width - side)
            y = min(iy * min_dist, canvas.height - side)
            
            # Check if this position is too close to any existing square
            too_close = False
            for existing in placed_squares:
                dx = (x + side/2) - (existing.x + existing.side/2)
                dy = (y + side/2) - (existing.y + existing.side/2)
                if math.sqrt(dx*dx + dy*dy) < min_dist:
                    too_close = True
                    break
            
            if not too_close:
                placed_squares.append(Square(int(x), int(y), int(side)))
        
        squares.extend(placed_squares)

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
        # Convert coordinates to integers
        top_left = (int(sq.x * scale), int(sq.y * scale))
        bottom_rt = (int((sq.x + sq.side) * scale), int((sq.y + sq.side) * scale))
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
        max_side_ratio=0.2,
        min_side_ratio=0.2,
        n_sizes=10,
        min_distance_ratio=1.4,  # Squares must be at least 10% of their size apart
        rng_seed=42,
    )
    print(f"Generated {len(sampled)} sample regions")
    visualise_sample(canvas, sampled, scale=1, show=True)
