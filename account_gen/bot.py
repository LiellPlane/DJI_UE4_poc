from interactions import WebInteraction, FontConfig
from vision import generate_string_pattern,display_image, find_searchpattern_scale, PatternMatch
def main():
    bot = WebInteraction(url="https://gtmetrix.com/")
    # calib_img, calib_str = bot.get_calibration_screenshot(url="https://old.reddit.com",save_to_disk=False)
    # str_pattern = generate_string_pattern("CALIBERT", FontConfig)
    text_img = bot.get_text_area_screenshot(url="https://gtmetrix.com/",text="COOKIE")
    screenshot, scale = bot.get_raw_screenshot()
    print(f"scale: {scale}")
    # display_image(text_img,wait_for_key=True)
    res: PatternMatch = find_searchpattern_scale(img=screenshot, pattern=text_img)
    display_image(res.debug_image,wait_for_key=True)
    print(res.score)

if __name__ == '__main__':
    main()
    