import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import UserAgent
import undetected_chromedriver as uc


class RedditBot:
    def __init__(self, base_url):
        self.base_url = base_url
        self.driver = None
        self.wait = None
        self.setup_driver()

    def setup_driver(self):
        """Setup the Chrome driver with human-like settings"""
        options = uc.ChromeOptions()
        options.add_argument(f"user-agent={UserAgent().random}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        self.driver = uc.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        self.driver.maximize_window()

    def human_like_delay(self, min_delay=1, max_delay=3):
        """Add random delay to simulate human behavior"""
        time.sleep(random.uniform(min_delay, max_delay))

    def human_like_typing(self, element, text):
        """Type text with human-like delays between keystrokes"""
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.1, 0.3))

    def move_to_element(self, element):
        """Move to element with human-like mouse movement"""
        actions = ActionChains(self.driver)
        actions.move_to_element(element)
        actions.perform()
        self.human_like_delay(0.5, 1.5)

    def create_account(self, username, email, password):
        """Create a new account with human-like behavior"""
        try:
            self.driver.get(f"{self.base_url}/register")
            self.human_like_delay()

            # Fill in registration form
            username_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            self.move_to_element(username_field)
            self.human_like_typing(username_field, username)

            email_field = self.driver.find_element(By.NAME, "email")
            self.move_to_element(email_field)
            self.human_like_typing(email_field, email)

            password_field = self.driver.find_element(By.NAME, "password")
            self.move_to_element(password_field)
            self.human_like_typing(password_field, password)

            # Submit form
            submit_button = self.driver.find_element(
                By.XPATH, "//button[@type='submit']"
            )
            self.move_to_element(submit_button)
            self.human_like_delay()
            submit_button.click()

            return True
        except Exception as e:
            print(f"Error creating account: {str(e)}")
            return False

    def create_post(self, subreddit, title, content):
        """Create a new post with human-like behavior"""
        try:
            self.driver.get(f"{self.base_url}/r/{subreddit}/submit")
            self.human_like_delay()

            # Fill in post form
            title_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "title"))
            )
            self.move_to_element(title_field)
            self.human_like_typing(title_field, title)

            content_field = self.driver.find_element(By.NAME, "content")
            self.move_to_element(content_field)
            self.human_like_typing(content_field, content)

            # Submit post
            submit_button = self.driver.find_element(
                By.XPATH, "//button[@type='submit']"
            )
            self.move_to_element(submit_button)
            self.human_like_delay()
            submit_button.click()

            return True
        except Exception as e:
            print(f"Error creating post: {str(e)}")
            return False

    def comment_on_post(self, post_url, comment_text):
        """Add a comment to a post with human-like behavior"""
        try:
            self.driver.get(post_url)
            self.human_like_delay()

            comment_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "comment"))
            )
            self.move_to_element(comment_field)
            self.human_like_typing(comment_field, comment_text)

            submit_button = self.driver.find_element(
                By.XPATH, "//button[@type='submit']"
            )
            self.move_to_element(submit_button)
            self.human_like_delay()
            submit_button.click()

            return True
        except Exception as e:
            print(f"Error commenting: {str(e)}")
            return False

    def vote_on_post(self, post_url, vote_type="up"):
        """Vote on a post with human-like behavior"""
        try:
            self.driver.get(post_url)
            self.human_like_delay()

            vote_button = self.wait.until(
                EC.presence_of_element_located(
                    (By.XPATH, f"//button[contains(@class, 'vote-{vote_type}')]")
                )
            )
            self.move_to_element(vote_button)
            self.human_like_delay()
            vote_button.click()

            return True
        except Exception as e:
            print(f"Error voting: {str(e)}")
            return False

    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
