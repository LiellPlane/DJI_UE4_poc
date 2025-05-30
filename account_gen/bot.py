from interactions import WebInteraction, FontConfig
from vision import (
    generate_string_pattern,
    display_image,
    find_searchpattern_scale,
    PatternMatchScale,
)
import cv2
import numpy as np
import time


def main():
    known_scale: int | None = None
    bot = WebInteraction(url="https://old.reddit.com")
    # calib_img, calib_str = bot.get_calibration_screenshot(url="https://old.reddit.com",save_to_disk=False)
    # str_pattern = generate_string_pattern("CALIBERT", FontConfig)

    # calibration stage
    text_img = bot.get_text_area_screenshot(
        url="https://old.reddit.com", text="BABY"
    )
    screenshot, scale = bot.get_raw_screenshot()

    # display_image(text_img,wait_for_key=True)
    res: PatternMatchScale = find_searchpattern_scale(
        img=screenshot, pattern=text_img, save_debug_imgs=True
    )
    known_scale = res.scale
    # display_image(res.debug_image,wait_for_key=True)

    res_scroll = True
    while True:
        screenshot, scale = bot.get_raw_screenshot()
        res_patmatch: PatternMatchScale = find_searchpattern_scale(
            img=screenshot,
            pattern=text_img,
            save_debug_imgs=True,
            known_scale=known_scale,  # we know the scale of the text area, so we can use it to find the search pattern and reduce seektime
        )

        # Draw the match results
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

        display_image(screenshot)
        if res_scroll is False:
            break
        res_scroll: bool = bot.scroll_page(direction="down", unit="half_page")
    # for testing - we know ABOUT is at the bottom of the page
    # lets see if we can click it using our new output
    bot.click_coordinate(
        x=res_patmatch.x + res_patmatch.width // 2,
        y=res_patmatch.y + res_patmatch.height // 2,
    )
    time.sleep(10)

    print(res_patmatch.score)


if __name__ == "__main__":
    main()
