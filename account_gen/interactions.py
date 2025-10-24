from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from fake_useragent import UserAgent
import time
import random
import logging
import cv2
import numpy as np
from io import BytesIO
import os
import base64
import json
from typing import Literal


class FontConfig:
    """Configuration for font settings used across the application"""

    FONT_FAMILY = "Arial, sans-serif"
    FONT_SIZE = "16px"
    TEXT_TRANSFORM = "uppercase"
    FONT_WEIGHT = "normal"
    FONT_STYLE = "normal"

    @classmethod
    def get_font_style_js(cls) -> str:
        """Returns JavaScript string for font styling"""
        return f"""
            fontFamily: '{cls.FONT_FAMILY}',
            fontSize: '{cls.FONT_SIZE}',
            textTransform: '{cls.TEXT_TRANSFORM}',
            fontWeight: '{cls.FONT_WEIGHT}',
            fontStyle: '{cls.FONT_STYLE}'
        """


class WebInteraction:
    def __init__(self, url):
        self.url = url
        self.driver = None
        self.wait = None
        self.setup_driver()
        self.normalize_page()

    def setup_driver(self):
        """Setup the Chrome driver with human-like settings"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                options = Options()
                options.add_argument(f"user-agent={UserAgent().random}")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-extensions")
                options.add_argument("--disable-notifications")
                options.add_argument("--start-maximized")
                options.add_argument("--log-level=3")  # Suppress logging
                options.add_argument("--silent")
                # Force 1:1 device pixel ratio
                options.add_argument("--force-device-scale-factor=1")
                options.add_experimental_option(
                    "excludeSwitches", ["enable-logging"]
                )  # Suppress DevTools logging

                self.driver = webdriver.Chrome(options=options)
                self.wait = WebDriverWait(self.driver, 10)
                self.driver.get(self.url)
                self.normalize_page()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(
                        f"Failed to setup Chrome driver after {max_retries} attempts: {str(e)}"
                    )

    def normalize_page(self):
        """Normalize all text on the page to make it easier for computer vision."""
        normalize_script = f"""
        document.body.style.fontFamily = '{FontConfig.FONT_FAMILY}';
        document.body.style.fontSize = '{FontConfig.FONT_SIZE}';
        document.body.style.textTransform = '{FontConfig.TEXT_TRANSFORM}';
        document.body.style.fontWeight = '{FontConfig.FONT_WEIGHT}';
        document.body.style.fontStyle = '{FontConfig.FONT_STYLE}';
        document.body.style.color = 'black';
        
        const elements = document.getElementsByTagName('*');
        for (let element of elements) {{
            element.style.fontFamily = '{FontConfig.FONT_FAMILY}';
            element.style.fontSize = '{FontConfig.FONT_SIZE}';
            element.style.textTransform = '{FontConfig.TEXT_TRANSFORM}';
            element.style.fontWeight = '{FontConfig.FONT_WEIGHT}';
            element.style.fontStyle = '{FontConfig.FONT_STYLE}';
            element.style.color = 'black';
        }}
        """
        self.driver.execute_script(normalize_script)

    def get_calibration_screenshot(
        self, url, save_to_disk=False, filename=None
    ) -> tuple[np.array, str]:
        """Load a URL, add calibration text, and return a screenshot for vision system calibration.

        Args:
            url (str): The URL to load for calibration
            save_to_disk (bool): Whether to save the screenshot to disk
            filename (str): Optional filename to save the screenshot as. If None and save_to_disk is True,
                          generates a timestamp-based filename.

        Returns:
            numpy.ndarray: Screenshot with calibration text, or None if failed


        """
        try:
            # Load the URL
            self.driver.get(url)
            self.wait_for_page_load()

            # Add calibration text with same font settings as normalize_page
            calibration_script = f"""
            const calibertDiv = document.createElement('div');
            calibertDiv.textContent = 'CALIBERT';
            calibertDiv.style.position = 'fixed';
            calibertDiv.style.top = '0';
            calibertDiv.style.left = '0';
            calibertDiv.style.zIndex = '9999';
            calibertDiv.style.backgroundColor = 'white';
            calibertDiv.style.padding = '5px';
            calibertDiv.style.fontFamily = '{FontConfig.FONT_FAMILY}';
            calibertDiv.style.fontSize = '{FontConfig.FONT_SIZE}';
            calibertDiv.style.textTransform = '{FontConfig.TEXT_TRANSFORM}';
            calibertDiv.style.fontWeight = '{FontConfig.FONT_WEIGHT}';
            calibertDiv.style.fontStyle = '{FontConfig.FONT_STYLE}';
            document.body.insertBefore(calibertDiv, document.body.firstChild);
            """
            self.driver.execute_script(calibration_script)

            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()
            image_np = cv2.imdecode(
                np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR
            )

            if save_to_disk:
                if filename is None:
                    filename = f"calibration_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, image_np)

            # Remove calibration text
            cleanup_script = """
            const calibertDiv = document.querySelector('div[style*="CALIBERT"]');
            if (calibertDiv) calibertDiv.remove();
            """
            self.driver.execute_script(cleanup_script)

            return image_np, "CALIBERT"

        except Exception as e:
            raise

    def get_raw_screenshot(
        self, save_to_disk=False, filename=None
    ) -> tuple[np.array, float]:
        """Get a raw screenshot of the current page that maintains coordinate consistency for mouse interactions.

        This method ensures that the screenshot coordinates will match what's used for mouse interactions
        by accounting for device pixel ratio and window scaling.

        Args:
            save_to_disk (bool): Whether to save the screenshot to disk
            filename (str): Optional filename to save the screenshot as. If None and save_to_disk is True,
                          generates a timestamp-based filename.

        Returns:
            tuple[np.array, float]: A tuple containing:
                - numpy.ndarray: The raw screenshot
                - float: The device pixel ratio (scale factor) that should be used to convert between
                        screenshot coordinates and actual mouse coordinates
        """
        try:
            # Get the device pixel ratio before taking screenshot
            scale = self.driver.execute_script("return window.devicePixelRatio;")

            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()
            image_np = cv2.imdecode(
                np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR
            )

            if save_to_disk:
                if filename is None:
                    filename = f"raw_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, image_np)

            return image_np, scale

        except Exception as e:
            raise Exception(f"Error in get_raw_screenshot: {str(e)}")

    def get_text_area_screenshot(
        self, text, url=None, save_to_disk=False, filename=None
    ) -> np.array:
        """Load a URL (if provided), add custom text, and return a screenshot of the area containing that text.

        Args:
            text (str): The text to display and capture
            url (str, optional): The URL to load. If None, uses the current page.
            save_to_disk (bool): Whether to save the screenshot to disk
            filename (str): Optional filename to save the screenshot as. If None and save_to_disk is True,
                          generates a timestamp-based filename.

        Returns:
            numpy.ndarray: Screenshot of the area containing the text
        """
        try:
            # Load the URL if provided
            if url is not None:
                self.driver.get(url)
                self.wait_for_page_load()

            # Create an isolated div
            create_div_script = f"""
            const textDiv = document.createElement('div');
            textDiv.textContent = '{text}';
            textDiv.style.position = 'fixed';
            textDiv.style.top = '50%';
            textDiv.style.left = '50%';
            textDiv.style.transform = 'translate(-50%, -50%)';
            textDiv.style.zIndex = '999999';
            textDiv.style.backgroundColor = 'white';
            textDiv.style.width = '1000px';
            textDiv.style.height = '500px';
            textDiv.style.display = 'flex';
            textDiv.style.alignItems = 'center';
            textDiv.style.justifyContent = 'center';
            textDiv.style.pointerEvents = 'none';
            textDiv.style.isolation = 'isolate';
            textDiv.style.willChange = 'transform';
            textDiv.style.backfaceVisibility = 'hidden';
            textDiv.style.transformStyle = 'preserve-3d';
            textDiv.style.fontFamily = '{FontConfig.FONT_FAMILY}';
            textDiv.style.fontSize = '{FontConfig.FONT_SIZE}';
            textDiv.style.textTransform = '{FontConfig.TEXT_TRANSFORM}';
            textDiv.style.fontWeight = '{FontConfig.FONT_WEIGHT}';
            textDiv.style.fontStyle = '{FontConfig.FONT_STYLE}';
            textDiv.style.color = 'black';
            textDiv.style.lineHeight = '1';
            textDiv.style.letterSpacing = 'normal';
            textDiv.style.whiteSpace = 'nowrap';
            document.body.appendChild(textDiv);
            
            // Get the dimensions
            const rect = textDiv.getBoundingClientRect();
            return {{
                width: Math.ceil(rect.width),
                height: Math.ceil(rect.height),
                left: Math.ceil(rect.left),
                top: Math.ceil(rect.top)
            }};
            """

            # Get the dimensions of the text div
            dimensions = self.driver.execute_script(create_div_script)

            # Wait a moment for the div to be fully rendered
            time.sleep(0.5)

            # Take screenshot of the entire page
            screenshot = self.driver.get_screenshot_as_png()
            image_np = cv2.imdecode(
                np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR
            )

            # Get the scale factor of the page
            scale_script = "return window.devicePixelRatio;"
            scale = self.driver.execute_script(scale_script)
            print(f"Device pixel ratio: {scale}")

            # Calculate initial crop coordinates
            x = int(dimensions["left"] * scale)
            y = int(dimensions["top"] * scale)
            w = int(dimensions["width"] * scale)
            h = int(dimensions["height"] * scale)
            print(
                f"Initial crop dimensions (physical pixels): x={x}, y={y}, w={w}, h={h}"
            )

            # Ensure we don't go out of bounds
            h_img, w_img = image_np.shape[:2]
            print(f"Image dimensions: {w_img}x{h_img}")
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            print(f"Adjusted crop dimensions: x={x}, y={y}, w={w}, h={h}")

            if w <= 0 or h <= 0:
                raise Exception(f"Invalid crop dimensions: x={x}, y={y}, w={w}, h={h}")

            # Initial crop
            cropped = image_np[y : y + h, x : x + w]
            print(f"Cropped image shape: {cropped.shape}")

            # Aggressive crop in from edges (50 CSS pixels worth)
            edge_margin = int(50 * scale)
            cropped = cropped[edge_margin:-edge_margin, edge_margin:-edge_margin]
            print(f"After aggressive crop shape: {cropped.shape}")

            # Now find the actual text area
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            coords = np.nonzero(binary)

            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()

                # Add small padding
                padding = 1
                y_min = max(0, y_min - padding)
                y_max = min(cropped.shape[0], y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(cropped.shape[1], x_max + padding)

                # Final crop to text area
                cropped = cropped[y_min:y_max, x_min:x_max]
                print(f"Final text area shape: {cropped.shape}")

            if save_to_disk:
                if filename is None:
                    filename = f"text_area_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, cropped)

            # Remove the div
            cleanup_script = """
            const textDiv = document.querySelector('div[style*="z-index: 9999"]');
            if (textDiv) textDiv.remove();
            """
            self.driver.execute_script(cleanup_script)

            return cropped

        except Exception as e:
            # Clean up div if it exists
            try:
                self.driver.execute_script("""
                    const textDiv = document.querySelector('div[style*="z-index: 9999"]');
                    if (textDiv) textDiv.remove();
                """)
            except:
                pass
            raise Exception(f"Error in get_text_area_screenshot: {str(e)}")

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

    def click_coordinate(self, x, y):
        """Click at specific coordinates with human-like behavior"""
        try:
            actions = ActionChains(self.driver)
            # Add some randomness to the click position
            x += random.uniform(-2, 2)
            y += random.uniform(-2, 2)
            actions.move_by_offset(x, y)
            self.human_like_delay(0.3, 0.7)
            actions.click()
            actions.perform()

            # Add visual indicator of click location
            indicator_script = f"""
            const indicator = document.createElement('div');
            indicator.style.position = 'fixed';
            indicator.style.left = '{x}px';
            indicator.style.top = '{y}px';
            indicator.style.width = '50px';
            indicator.style.height = '50px';
            indicator.style.backgroundColor = 'red';
            indicator.style.borderRadius = '50%';
            indicator.style.opacity = '0.7';
            indicator.style.pointerEvents = 'none';
            indicator.style.zIndex = '999999';
            document.body.appendChild(indicator);
            
            // Remove indicator after 2 seconds
            setTimeout(() => {{
                indicator.remove();
            }}, 2000);
            """
            self.driver.execute_script(indicator_script)

            # Ensure page remains normalised after click
            self.normalize_page()

            return True
        except Exception as e:
            print(f"Error clicking coordinate: {str(e)}")
            return False

    def scroll_page(
        self, amount=None, direction: Literal["down", "up"] = "down", unit="half_page"
    ):
        """Scroll the page with human-like behavior

        Args:
            amount: The amount to scroll. If None, uses a random amount.
            direction: 'up' or 'down'
            unit: 'pixels', 'half_page', or 'full_page'

        Returns:
            bool: True if the page was successfully scrolled, False if:
                - An error occurred during scrolling
                - The page could not be scrolled further in the requested direction
        """
        try:
            # Get current scroll position
            current_scroll = self.driver.execute_script("return window.pageYOffset;")

            if amount is None:
                if unit == "pixels":
                    amount = random.randint(300, 700)
                elif unit == "half_page":
                    amount = 0.5
                else:  # full_page
                    amount = 1.0

            # Get viewport height
            viewport_height = self.driver.execute_script("return window.innerHeight")

            # Calculate scroll amount based on unit
            if unit == "pixels":
                scroll_amount = amount
            else:  # half_page or full_page
                scroll_amount = int(viewport_height * amount)

            if direction == "up":
                scroll_amount = -scroll_amount

            # Scroll with smooth behavior
            self.driver.execute_script(
                f"window.scrollBy({{top: {scroll_amount}, behavior: 'smooth'}});"
            )
            self.human_like_delay(0.5, 1.5)

            # Get new scroll position
            new_scroll = self.driver.execute_script("return window.pageYOffset;")

            # Check if we actually scrolled
            if direction == "down" and new_scroll <= current_scroll:
                return False  # Couldn't scroll down further
            elif direction == "up" and new_scroll >= current_scroll:
                return False  # Couldn't scroll up further

            return True
        except Exception as e:
            print(f"Error scrolling page: {str(e)}")
            return False

    def enter_text(self, element, text):
        """Enter text into an element with human-like behavior"""
        try:
            self.move_to_element(element)
            self.human_like_typing(element, text)
            return True
        except Exception as e:
            print(f"Error entering text: {str(e)}")
            return False

    def get_screen(self, save_to_disk=False, filename=None):
        """Get screenshot of current page, optionally save to disk"""
        try:
            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()

            # Convert to numpy array for processing
            image_np = cv2.imdecode(
                np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR
            )

            if save_to_disk:
                if filename is None:
                    filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, image_np)

            return image_np
        except Exception as e:
            print(f"Error taking screenshot: {str(e)}")
            return None

    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()

    def hover_over(self, element):
        """Hover over an element with human-like behavior"""
        try:
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            actions.perform()
            self.human_like_delay(0.5, 1.5)
            return True
        except Exception as e:
            print(f"Error hovering over element: {str(e)}")
            return False

    def drag_and_drop(self, source_element, target_element):
        """Drag and drop with human-like behavior"""
        try:
            actions = ActionChains(self.driver)
            actions.click_and_hold(source_element)
            self.human_like_delay(0.3, 0.7)
            actions.move_to_element(target_element)
            self.human_like_delay(0.3, 0.7)
            actions.release()
            actions.perform()
            return True
        except Exception as e:
            print(f"Error in drag and drop: {str(e)}")
            return False

    def select_dropdown_option(self, dropdown_element, option_text):
        """Select an option from a dropdown with human-like behavior"""
        try:
            self.move_to_element(dropdown_element)
            dropdown_element.click()
            self.human_like_delay(0.3, 0.7)

            # Find and click the option
            option = self.wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, f"//option[contains(text(), '{option_text}')]")
                )
            )
            self.move_to_element(option)
            option.click()
            return True
        except Exception as e:
            print(f"Error selecting dropdown option: {str(e)}")
            return False

    def check_checkbox(self, checkbox_element, should_check=True):
        """Check or uncheck a checkbox with human-like behavior"""
        try:
            self.move_to_element(checkbox_element)
            if checkbox_element.is_selected() != should_check:
                checkbox_element.click()
            return True
        except Exception as e:
            print(f"Error toggling checkbox: {str(e)}")
            return False

    def upload_file(self, file_input_element, file_path):
        """Upload a file with human-like behavior"""
        try:
            self.move_to_element(file_input_element)
            file_input_element.send_keys(file_path)
            self.human_like_delay(1, 2)  # Longer delay for file upload
            return True
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return False

    def press_key(self, key):
        """Press a keyboard key with human-like behavior"""
        try:
            actions = ActionChains(self.driver)
            actions.send_keys(key)
            self.human_like_delay(0.1, 0.3)
            actions.perform()
            return True
        except Exception as e:
            print(f"Error pressing key: {str(e)}")
            return False

    def wait_for_element(self, locator, timeout=10):
        """Wait for an element to be present and visible"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located(locator)
            )
            return element
        except Exception as e:
            print(f"Error waiting for element: {str(e)}")
            return None

    def wait_for_page_load(self):
        """Wait for page to load and normalize it"""
        try:
            self.wait.until(
                lambda driver: driver.execute_script("return document.readyState")
                == "complete"
            )
            self.normalize_page()
        except Exception as e:
            print(f"Error waiting for page load: {str(e)}")

    def refresh_page(self):
        """Refresh the page with human-like behavior"""
        try:
            self.driver.refresh()
            self.human_like_delay(1, 2)
            self.wait_for_page_load()
            return True
        except Exception as e:
            print(f"Error refreshing page: {str(e)}")
            return False

    def go_back(self):
        """Go back to previous page with human-like behavior"""
        try:
            self.driver.back()
            self.human_like_delay(1, 2)
            self.wait_for_page_load()
            return True
        except Exception as e:
            print(f"Error going back: {str(e)}")
            return False

    def go_forward(self):
        """Go forward to next page with human-like behavior"""
        try:
            self.driver.forward()
            self.human_like_delay(1, 2)
            self.wait_for_page_load()
            return True
        except Exception as e:
            print(f"Error going forward: {str(e)}")
            return False

    def clear_text(self, element):
        """Clear text from an input field with human-like behavior"""
        try:
            self.move_to_element(element)
            element.clear()
            self.human_like_delay(0.3, 0.7)
            return True
        except Exception as e:
            print(f"Error clearing text: {str(e)}")
            return False

    def double_click(self, element):
        """Double click an element with human-like behavior"""
        try:
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            self.human_like_delay(0.3, 0.7)
            actions.double_click()
            actions.perform()
            return True
        except Exception as e:
            print(f"Error double clicking: {str(e)}")
            return False

    def right_click(self, element):
        """Right click an element with human-like behavior"""
        try:
            actions = ActionChains(self.driver)
            actions.move_to_element(element)
            self.human_like_delay(0.3, 0.7)
            actions.context_click()
            actions.perform()
            return True
        except Exception as e:
            print(f"Error right clicking: {str(e)}")
            return False

    def get_scroll_position(self) -> int:
        """Get the current scroll position in pixels from the top of the page.

        Returns:
            int: The number of pixels scrolled from the top of the page
        """
        try:
            return self.driver.execute_script("return window.pageYOffset;")
        except Exception as e:
            print(f"Error getting scroll position: {str(e)}")
            return 0

    def scroll_to_position(self, offset: int) -> bool:
        """Scroll to a specific absolute position on the page using successive 
        approximation for exact positioning.

        Args:
            offset (int): The absolute position in pixels from the top of the 
                         page to scroll to

        Returns:
            bool: True if successfully scrolled to exact position, False otherwise
        """
        try:
            target_position = offset
            max_attempts = 10
            final_tolerance = 2  # Final precision within 2 pixels
            
            for attempt in range(max_attempts):
                current_position = self.get_scroll_position()
                difference = target_position - current_position
                
                # If we're within final tolerance, we're done
                if abs(difference) <= final_tolerance:
                    return True
                
                # Determine scroll behaviour based on distance
                if abs(difference) > 100:
                    # Large distance - use smooth scrolling directly to target
                    scroll_behaviour = "smooth"
                    scroll_target = target_position
                    delay = 0.8  # Longer delay for smooth scroll
                elif abs(difference) > 20:
                    # Medium distance - scroll most of the way
                    scroll_behaviour = "auto"
                    scroll_target = current_position + int(difference * 0.8)
                    delay = 0.3
                else:
                    # Small distance - precise pixel-by-pixel adjustment
                    scroll_behaviour = "auto"
                    scroll_target = target_position
                    delay = 0.2
                
                # Execute the scroll
                self.driver.execute_script(
                    f"window.scrollTo({{top: {scroll_target}, "
                    f"behavior: '{scroll_behaviour}'}});"
                )
                
                # Wait for scroll to complete
                self.human_like_delay(delay, delay + 0.2)
                
                # Check if we've made progress
                new_position = self.get_scroll_position()
                if new_position == current_position and abs(difference) > final_tolerance:
                    # No movement and still not at target - might be at page limit
                    return False
            
            # Final check after all attempts
            final_position = self.get_scroll_position()
            return abs(final_position - target_position) <= final_tolerance
            
        except Exception as e:
            print(f"Error scrolling to position: {str(e)}")
            return False
