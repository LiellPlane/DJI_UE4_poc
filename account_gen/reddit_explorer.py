from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_vision.log'),
        logging.StreamHandler()
    ]
)

def normalize_page_text(driver):
    """Normalize all text on the page to make it easier for computer vision."""
    normalize_script = """
    document.body.style.fontFamily = 'Arial, sans-serif';
    document.body.style.fontSize = '16px';
    document.body.style.textTransform = 'uppercase';
    
    // Apply to all elements
    const elements = document.getElementsByTagName('*');
    for (let element of elements) {
        element.style.fontFamily = 'Arial, sans-serif';
        element.style.fontSize = '16px';
        element.style.textTransform = 'uppercase';
    }
    """
    driver.execute_script(normalize_script)
    logging.info("Page text normalized")

def explore_reddit():
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_argument('--start-maximized')
    
    try:
        # Initialize the driver with our options
        logging.info("Initializing Chrome...")
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 10)
        
        # Navigate to old Reddit interface
        logging.info("Navigating to Reddit...")
        driver.get("https://old.reddit.com")
        time.sleep(3)
        
        # Normalize text on the page
        normalize_page_text(driver)
        
        # Keep the browser open for examination
        logging.info("Page loaded and normalized. Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        logging.info("Exiting...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    explore_reddit() 