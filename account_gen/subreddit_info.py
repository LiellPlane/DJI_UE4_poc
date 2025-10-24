from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import csv
from datetime import datetime
import re


def get_subreddit_info():
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")

    try:
        # Initialize the driver
        print("Initializing Chrome...")
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 10)

        # List of subreddits to check
        subreddits = [
            "cscareerquestionsuk",
            "cscareerquestions",
            "programming",
            "learnprogramming",
            "coding",
            "webdev",
            "python",
            "javascript",
            "java",
            "cpp",
        ]

        # Create CSV file for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"subreddit_info_{timestamp}.csv"

        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Subreddit",
                    "Subscribers",
                    "Active Users",
                    "Created Date",
                    "Description",
                    "Post Frequency",
                ]
            )

            for subreddit in subreddits:
                try:
                    print(f"\nFetching info for r/{subreddit}...")

                    # Navigate to subreddit
                    driver.get(f"https://old.reddit.com/r/{subreddit}/")
                    time.sleep(2)

                    # Get subscriber count
                    try:
                        subscribers_text = driver.find_element(
                            By.CSS_SELECTOR, "span.subscribers"
                        ).text
                        subscribers = int(re.sub(r"[^\d]", "", subscribers_text))
                    except:
                        subscribers = "N/A"

                    # Get active users
                    try:
                        active_text = driver.find_element(
                            By.CSS_SELECTOR, "p.users-online"
                        ).text
                        active_users = int(re.sub(r"[^\d]", "", active_text))
                    except:
                        active_users = "N/A"

                    # Get description
                    try:
                        description = (
                            driver.find_element(By.CSS_SELECTOR, "div.md")
                            .text.replace("\n", " ")
                            .strip()
                        )
                    except:
                        description = "N/A"

                    # Get creation date (approximate from oldest post)
                    try:
                        driver.get(f"https://old.reddit.com/r/{subreddit}/new/")
                        time.sleep(2)
                        oldest_post = driver.find_elements(By.CSS_SELECTOR, "time")[-1]
                        created_date = oldest_post.get_attribute("datetime")[
                            :10
                        ]  # Get YYYY-MM-DD
                    except:
                        created_date = "N/A"

                    # Calculate post frequency
                    try:
                        posts = driver.find_elements(By.CSS_SELECTOR, "div.thing.link")
                        post_frequency = f"{len(posts)} posts visible"
                    except:
                        post_frequency = "N/A"

                    # Write to CSV
                    writer.writerow(
                        [
                            subreddit,
                            subscribers,
                            active_users,
                            created_date,
                            description,
                            post_frequency,
                        ]
                    )

                    print(f"Subscribers: {subscribers:,}")
                    print(f"Active Users: {active_users:,}")
                    print(f"Created: {created_date}")
                    print(f"Description: {description[:100]}...")
                    print(f"Post Frequency: {post_frequency}")

                    # Be nice to Reddit's servers
                    time.sleep(2)

                except Exception as e:
                    print(f"Error fetching data for r/{subreddit}: {str(e)}")
                    writer.writerow(
                        [subreddit, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"]
                    )

        print(f"\nResults saved to {csv_filename}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        try:
            time.sleep(2)
            driver.quit()
        except:
            pass


if __name__ == "__main__":
    get_subreddit_info()
