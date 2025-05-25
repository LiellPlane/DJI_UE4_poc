from interactions import WebInteraction, FontConfig
from vision import generate_string_pattern,display_image
def main():
    bot = WebInteraction(url="https://old.reddit.com")
    calib_img, calib_str = bot.get_calibration_screenshot(url="https://old.reddit.com",save_to_disk=False)
    # str_pattern = generate_string_pattern("CALIBERT", FontConfig)
    tet_img = bot.get_text_area_screenshot(url="https://old.reddit.com",text="MY SUBREDDITS")
    # display_image(tet_img,wait_for_key=True)

if __name__ == '__main__':
    main()
    