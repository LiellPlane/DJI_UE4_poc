import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import csv
import random
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reddit_scraper.log"), logging.StreamHandler()],
)


def get_subreddit_list():
    driver = None
    try:
        # Setup Chrome options
        options = uc.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")

        # Get Chrome binary path
        chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if not os.path.exists(chrome_path):
            chrome_path = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

        # Initialize the driver with explicit binary path
        logging.info("Initializing Chrome...")
        driver = uc.Chrome(
            options=options,
            browser_executable_path=chrome_path,
            version_main=122,  # Adjust this to match your Chrome version
            driver_executable_path=None,  # Let it auto-download
        )
        wait = WebDriverWait(driver, 20)

        # Dictionary to store subreddits and their subscriber counts
        subreddits = {}

        # List of different subreddit discovery pages
        discovery_pages = [
            "https://old.reddit.com/subreddits/popular/",
            "https://old.reddit.com/subreddits/new/",
            "https://old.reddit.com/subreddits/rising/",
            "https://old.reddit.com/subreddits/trending/",
        ]

        # Fixed filename for output
        csv_filename = "subreddit_list.csv"

        for base_url in discovery_pages:
            page_count = 0
            max_pages = 3  # Reduced pages per category to get more variety

            while page_count < max_pages:
                url = f"{base_url}?count={page_count * 25}"
                logging.info(f"Fetching page {page_count + 1} from {base_url}...")
                logging.info(f"URL: {url}")

                try:
                    logging.info("Loading page...")
                    driver.get(url)
                    time.sleep(random.uniform(3, 5))

                    logging.info("Waiting for content to load...")
                    wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.thing"))
                    )

                    logging.info("Finding subreddit entries...")
                    entries = driver.find_elements(By.CSS_SELECTOR, "div.thing")
                    logging.info(f"Found {len(entries)} entries on page")

                    if not entries:
                        logging.warning("No more subreddits found in this category.")
                        break

                    for i, entry in enumerate(entries, 1):
                        try:
                            logging.info(f"Processing entry {i}/{len(entries)}")

                            # Get subreddit name
                            logging.info("Looking for subreddit link...")
                            subreddit_link = entry.find_element(
                                By.CSS_SELECTOR, "a.title"
                            )
                            subreddit = subreddit_link.text
                            if subreddit.startswith("r/"):
                                subreddit = subreddit[2:]
                            logging.info(f"Found subreddit: r/{subreddit}")

                            # Visit the subreddit page to get subscriber count
                            subreddit_url = f"https://old.reddit.com/r/{subreddit}/"
                            logging.info(f"Visiting subreddit page: {subreddit_url}")
                            driver.get(subreddit_url)
                            time.sleep(random.uniform(2, 3))

                            try:
                                logging.info("Looking for subscriber count...")
                                subscribers_text = driver.find_element(
                                    By.CSS_SELECTOR, "span.subscribers"
                                ).text
                                subscribers = int(
                                    "".join(filter(str.isdigit, subscribers_text))
                                )
                                logging.info(f"Found {subscribers:,} subscribers")
                            except NoSuchElementException:
                                logging.warning(
                                    "Could not find subscriber count, defaulting to 0"
                                )
                                subscribers = 0

                            # Only add if it's not already in our list
                            if subreddit not in subreddits:
                                subreddits[subreddit] = subscribers
                                logging.info(
                                    f"Added r/{subreddit} with {subscribers:,} subscribers"
                                )

                                # Save progress after each new subreddit
                                logging.info("Saving progress to CSV...")
                                with open(
                                    csv_filename, "w", newline="", encoding="utf-8"
                                ) as csvfile:
                                    writer = csv.writer(csvfile)
                                    writer.writerow(["Subreddit", "Subscribers"])
                                    for sub, sub_count in sorted(
                                        subreddits.items(),
                                        key=lambda x: x[1],
                                        reverse=True,
                                    ):
                                        writer.writerow([sub, sub_count])

                                logging.info(
                                    f"Progress: Found {len(subreddits)} unique subreddits so far"
                                )

                        except Exception as e:
                            logging.error(f"Error processing entry: {str(e)}")
                            continue

                    page_count += 1

                except TimeoutException:
                    logging.error("Timeout waiting for page to load. Retrying...")
                    time.sleep(5)
                    continue
                except Exception as e:
                    logging.error(f"Error processing page: {str(e)}")
                    time.sleep(5)
                    continue

        logging.info(f"Final Results:")
        logging.info(f"Total subreddits found: {len(subreddits)}")
        logging.info("Top 10 subreddits by subscriber count:")
        for subreddit, subscribers in sorted(
            subreddits.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            logging.info(f"r/{subreddit}: {subscribers:,} subscribers")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    finally:
        if driver:
            logging.info("Closing Chrome...")
            try:
                driver.quit()
            except Exception as e:
                logging.error(f"Error closing Chrome: {str(e)}")


if __name__ == "__main__":
    get_subreddit_list()
