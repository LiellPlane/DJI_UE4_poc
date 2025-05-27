from interactions import WebInteraction, FontConfig
from vision import generate_string_pattern,display_image, find_searchpattern_scale, PatternMatch
def main():
    bot = WebInteraction(url="https://old.reddit.com")
    # calib_img, calib_str = bot.get_calibration_screenshot(url="https://old.reddit.com",save_to_disk=False)
    # str_pattern = generate_string_pattern("CALIBERT", FontConfig)

    # calibration stage
    text_img = bot.get_text_area_screenshot(url="https://old.reddit.com",text="ABOUT")
    screenshot, scale = bot.get_raw_screenshot()
    print(f"scale: {scale}")
    # display_image(text_img,wait_for_key=True)
    res: PatternMatch = find_searchpattern_scale(img=screenshot, pattern=text_img,save_debug_imgs=True)
    # display_image(res.debug_image,wait_for_key=True)

    res_scroll = True
    while True:
        screenshot, scale = bot.get_raw_screenshot()
        res_patmatch: PatternMatch = find_searchpattern_scale(
            img=screenshot,
            pattern=text_img,
            save_debug_imgs=True
            )
        # cv2.circle(screenshot, (res_patmatch.x, res_patmatch.y), 5, (0, 0, 255), -1)
        display_image(screenshot)
        if res_scroll is False:
            break
        res_scroll: bool = bot.scroll_page(direction="down", unit="half_page")
    print(res_patmatch.score)

if __name__ == '__main__':
    main()
    