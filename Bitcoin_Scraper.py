import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from datetime import datetime


# Setup the Selenium WebDriver with the necessary options
def setup_driver():
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    ua = UserAgent()
    options.add_argument(f"user-agent={ua.random}")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_bitcoin_data(url):
    driver = setup_driver()
    driver.get(url)
    time.sleep(5)  # Wait for the page to load completely

    data = {}
    try:
        # Time of Scraping
        data["Time of Scraping"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Price (Current)
        try:
            # Use the first occurrence of an element with data-converter-target="price"
            current_price = driver.find_element(By.XPATH,
                "//div[@data-coin-show-target='staticCoinPrice']//span[@data-converter-target='price']").text.strip()
        except Exception as e:
            current_price = "N/A"
        data["Price"] = current_price

        # 24h Range (High/Low)
        try:
            # Locate the container with the text "24h Range" among its children
            range_container = driver.find_element(By.XPATH,
                                                  '//div[contains(@class, "tw-flex tw-justify-between") and contains(., "24h Range")]')
            # Find all spans inside this container with attribute data-price-target="price"
            range_spans = range_container.find_elements(By.XPATH, './/span[@data-price-target="price"]')
            if len(range_spans) >= 2:
                low_range = range_spans[0].text.strip()
                high_range = range_spans[1].text.strip()
                range_24h = f"{low_range} - {high_range}"
            else:
                range_24h = "N/A"
        except Exception as e:
            range_24h = "N/A"
        data["24h Range (High/Low)"] = range_24h

        # Market Cap
        try:
            market_cap = driver.find_element(
                By.XPATH,
                '//tr[.//th[contains(text(),"Market Cap")]]//span[@data-price-target="price"]'
            ).text.strip()
        except Exception as e:
            market_cap = "N/A"
        data["Market Cap"] = market_cap

        # 24h Trading Volume
        try:
            volume_24h = driver.find_element(
                By.XPATH,
                '//tr[.//th[contains(text(),"24 Hour Trading Vol")]]//span[@data-price-target="price"]'
            ).text.strip()
        except Exception as e:
            volume_24h = "N/A"
        data["24h Trading Volume"] = volume_24h

        # Circulating Supply
        try:
            circulating_supply = driver.find_element(
                By.XPATH,
                '//tr[.//th[contains(text(),"Circulating Supply")]]/td'
            ).text.strip()
        except Exception as e:
            circulating_supply = "N/A"
        data["Circulating Supply"] = circulating_supply

        # Price Change Percentages (from the price changes table)
        try:
            change_tds = driver.find_elements(By.XPATH, '//table//tbody/tr/td')
            if len(change_tds) >= 3:
                change_1h = change_tds[0].text.strip()
                change_24h = change_tds[1].text.strip()
                change_7d = change_tds[2].text.strip()
            else:
                change_1h = change_24h = change_7d = "N/A"
        except Exception as e:
            change_1h = change_24h = change_7d = "N/A"
        data["1h Price Change %"] = change_1h
        data["24h Price Change %"] = change_24h
        data["7d Price Change %"] = change_7d

        # Print extracted data for debugging
        for key, value in data.items():
            print(f"{key}: {value}")

        save_to_csv(data)
    except Exception as e:
        print(f"Error scraping data: {e}")
    finally:
        driver.quit()


def save_to_csv(data):
    file_name = "bitcoin_data.csv"
    header = [
        "Time of Scraping", "Price", "24h Range (High/Low)",
        "Market Cap", "24h Trading Volume", "Circulating Supply",
        "1h Price Change %", "24h Price Change %", "7d Price Change %"
    ]
    file_exists = False
    try:
        with open(file_name, mode='r', newline='', encoding='utf-8') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(file_name, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([data.get(col, "N/A") for col in header])


if __name__ == "__main__":
    url = "https://www.coingecko.com/en/coins/bitcoin"
    scrape_bitcoin_data(url)
