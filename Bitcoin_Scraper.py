import time
import csv
import os
import logging
from datetime import datetime
from typing import Dict
import schedule

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("bitcoin_scraper.log"),
        logging.StreamHandler()
    ]
)

def setup_driver() -> webdriver.Chrome:
    """Initialize Selenium WebDriver with custom options."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    ua = UserAgent()
    options.add_argument(f"user-agent={ua.random}")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def get_lebanon_time() -> str:
    """Return current time in Lebanon's timezone."""
    now_utc = datetime.now(pytz.utc)
    lebanon_time = now_utc.astimezone(pytz.timezone('Asia/Beirut'))
    return lebanon_time.strftime("%Y-%m-%d %H:%M:%S")


def save_to_csv(data: Dict[str, str], filename: str = "bitcoin_data.csv"):
    """Save scraped data to a CSV file."""
    folder = "data"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    header = list(data.keys())

    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([data.get(col, "N/A") for col in header])


def scrape_bitcoin_data(url: str):
    """Main scraping function."""
    driver = setup_driver()

    try:
        logging.info(f"Accessing {url}")
        driver.get(url)
        time.sleep(5)

        data = {"Time of Scraping": get_lebanon_time()}

        try:
            data["Price"] = driver.find_element(
                By.XPATH, "//div[@data-coin-show-target='staticCoinPrice']//span[@data-converter-target='price']"
            ).text.strip()
        except:
            data["Price"] = "N/A"

        try:
            range_container = driver.find_element(
                By.XPATH, '//div[contains(@class, "tw-flex tw-justify-between") and contains(., "24h Range")]'
            )
            spans = range_container.find_elements(By.XPATH, './/span[@data-price-target="price"]')
            if len(spans) >= 2:
                data["24h Range (High/Low)"] = f"{spans[0].text.strip()} - {spans[1].text.strip()}"
            else:
                data["24h Range (High/Low)"] = "N/A"
        except:
            data["24h Range (High/Low)"] = "N/A"

        try:
            data["Market Cap"] = driver.find_element(
                By.XPATH, '//tr[.//th[contains(text(),"Market Cap")]]//span[@data-price-target="price"]'
            ).text.strip()
        except:
            data["Market Cap"] = "N/A"

        try:
            data["24h Trading Volume"] = driver.find_element(
                By.XPATH, '//tr[.//th[contains(text(),"24 Hour Trading Vol")]]//span[@data-price-target="price"]'
            ).text.strip()
        except:
            data["24h Trading Volume"] = "N/A"

        try:
            data["Circulating Supply"] = driver.find_element(
                By.XPATH, '//tr[.//th[contains(text(),"Circulating Supply")]]/td'
            ).text.strip()
        except:
            data["Circulating Supply"] = "N/A"

        try:
            change_tds = driver.find_elements(By.XPATH, '//table//tbody/tr/td')
            data["1h Price Change %"] = change_tds[0].text.strip() if len(change_tds) >= 1 else "N/A"
            data["24h Price Change %"] = change_tds[1].text.strip() if len(change_tds) >= 2 else "N/A"
            data["7d Price Change %"] = change_tds[2].text.strip() if len(change_tds) >= 3 else "N/A"
        except:
            data["1h Price Change %"] = data["24h Price Change %"] = data["7d Price Change %"] = "N/A"

        logging.info("Scraped Data:")
        for key, value in data.items():
            logging.info(f"{key}: {value}")

        save_to_csv(data)

    except Exception as e:
        logging.error(f"Error during scraping: {e}")
    finally:
        driver.quit()

def job():
    scrape_bitcoin_data("https://www.coingecko.com/en/coins/bitcoin")

schedule.every(2).minutes.do(job)


if __name__ == "__main__":
    job()  # Run immediately once
    while True:
        schedule.run_pending()
        time.sleep(60)