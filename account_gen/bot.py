from interactions import WebInteraction, FontConfig
from vision import (
    generate_string_pattern,
    display_image,
    find_searchpattern_scale,
    PatternMatchScale,
    PatternMatchScroll,
)
import cv2
import numpy as np
import time


def get_text_matches_in_page(
    bot: WebInteraction, text: str, known_scale: int | None = None
) -> list[PatternMatchScale]:
    """
    get all matches from scrolling up and down the page
    """
    # first - ensure we are at the top of the page
    res_scroll = True
    while True:
        res_scroll: bool = bot.scroll_page(direction="up", unit="full_page")
        if res_scroll is False:
            break
    # then - scroll to bottom of page collecting matches as we go
    matches: list[PatternMatchScale] = []
    text_img = bot.get_text_area_screenshot(text=text)
    while True:
        screenshot, scale = bot.get_raw_screenshot()
        res_patmatch = PatternMatchScroll(**find_searchpattern_scale(
            img=screenshot,
            pattern=text_img,
            save_debug_imgs=True,
            known_scale=known_scale
        ).__dict__ | {"scroll_position_Y": bot.get_scroll_position()})

        # Draw the match results
        screenshot_annotated = draw_match_results(screenshot, res_patmatch, scale)
        display_image(screenshot_annotated)

        matches.append(res_patmatch)

        res_scroll: bool = bot.scroll_page(direction="down", unit="half_page")
        if res_scroll is False:
            break
    return matches

def draw_match_results(
    screenshot: np.ndarray, res_patmatch, scale: float
) -> np.ndarray:
    """Draw match results on a screenshot.

    Args:
        screenshot (np.ndarray): The screenshot to draw on
        res_patmatch: The pattern match result containing x, y, width, height and score
        scale (float): The scale factor to apply to coordinates

    Returns:
        np.ndarray: The screenshot with match results drawn on it
    """
    if res_patmatch.score > 0:
        # Scale the coordinates and dimensions
        x = int(res_patmatch.x * scale)
        y = int(res_patmatch.y * scale)
        w = int(res_patmatch.width * scale)
        h = int(res_patmatch.height * scale)

        # Draw rectangle around match
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw centre point
        cv2.circle(screenshot, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)
        # Add score text
        cv2.putText(
            screenshot,
            f"Score: {res_patmatch.score:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return screenshot


def main():
    known_scale: int | None = None
    bot = WebInteraction(url="https://old.reddit.com")
    # calib_img, calib_str = bot.get_calibration_screenshot(url="https://old.reddit.com",save_to_disk=False)
    # str_pattern = generate_string_pattern("CALIBERT", FontConfig)


    # CALIBRATION STAGE
    text_img = bot.get_text_area_screenshot(url="https://old.reddit.com", text="NONMATCHYSTRING")
    screenshot, scale = bot.get_raw_screenshot()

    # display_image(text_img,wait_for_key=True)
    res: PatternMatchScale = find_searchpattern_scale(
        img=screenshot, pattern=text_img, save_debug_imgs=True
    )
    known_scale = res.scale
    # display_image(res.debug_image,wait_for_key=True)

    res = get_text_matches_in_page(bot, text="COMMENTS", known_scale=known_scale)
    # test scrolling back to position works
    bot.scroll_to_position(offset=res[2].scroll_position_Y)
    bot.click_coordinate(x=res[2].x, y=res[2].y)
    time.sleep(3) #TODO can this be done better
    # res = get_text_matches_in_page(bot, text="REPORT", known_scale=known_scale)
    res = get_text_matches_in_page(bot, text="REPLY", known_scale=known_scale)
    bot.scroll_to_position(offset=res[5].scroll_position_Y)
    bot.click_coordinate(x=res[5].x, y=res[5].y)
    time.sleep(100)
    # print(res_patmatch.score)


if __name__ == "__main__":
    main()
