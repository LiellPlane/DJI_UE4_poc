from interactions import WebInteraction, FontConfig
from vision import generate_string_pattern
def main():
    bot = WebInteraction(url="https://old.reddit.com")
    calib_img, calib_str = bot.get_calibration_screenshot(url="https://old.reddit.com",save_to_disk=True)
    generate_string_pattern("wee wee", FontConfig)
if __name__ == '__main__':
    main()
    