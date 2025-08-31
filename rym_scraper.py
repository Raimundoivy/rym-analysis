import pandas as pd
import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
# New imports for explicit waits
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

# --- Configuration ---
YEARS_TO_SCRAPE = range(2000, 2026) 
BASE_URL = "https://rateyourmusic.com/charts/top/album/"

# --- Selenium Setup (Same as before) ---
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

service = ChromeService(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# --- Main Script ---
all_albums_data = []
print("Starting the scraping process with Selenium and Explicit Waits...")

for year in YEARS_TO_SCRAPE:
    target_url = f"{BASE_URL}{year}/"
    print(f"--- Scraping Top Albums for {year} from {target_url} ---")
    
    try:
        driver.get(target_url)
        
        # --- KEY CHANGE: Use an Explicit Wait instead of a static sleep ---
        # Wait up to 30 seconds for the first album container to be present.
        # This is the element we need for our data.
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "chart_item_container"))
        )
        
        # Once the wait is successful, we know the content is loaded.
        page_source = driver.page_source
        
    except TimeoutException:
        print(f"Content did not load for year {year} within 30 seconds. Skipping.")
        continue
    except Exception as e:
        print(f"An unexpected error occurred for year {year}. Error: {e}")
        continue

    # The parsing logic remains identical
    soup = BeautifulSoup(page_source, 'html.parser')
    album_containers = soup.find_all('div', class_='chart_item_container') 

    if not album_containers:
        # This check is now less likely to fail, but we keep it as a safeguard.
        print(f"Containers were located, but parsing failed for {year}.")
        continue

    for container in album_containers:
        try:
            title = container.find('a', class_='chart_item_title').text.strip()
            artist = container.find('a', class_='chart_item_artist').text.strip()
            avg_rating = container.find('span', class_='chart_item_avg_rating').text.strip()
            num_ratings = container.find('span', class_='chart_item_num_ratings').text.strip()
            release_date = container.find('span', class_='chart_item_date').text.strip()

            album_dict = {
                'year_charted': year,
                'title': title,
                'artist': artist,
                'avg_rating': avg_rating,
                'num_ratings': num_ratings,
                'release_date': release_date
            }
            all_albums_data.append(album_dict)
        except AttributeError:
            print(f"Skipping an entry in {year} due to missing data.")
            continue
    
    # We still keep our polite delay between different pages
    sleep_time = random.uniform(5, 15) # Can reduce this delay slightly now
    print(f"Successfully scraped {len(album_containers)} albums for {year}. Waiting for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)

driver.quit()
print("Scraping finished and browser closed.")

df = pd.DataFrame(all_albums_data)
raw_filename = 'rym_top_albums_2000-2025_raw.csv'
df.to_csv(raw_filename, index=False, encoding='utf-8')

print(f"\nSuccessfully saved raw data to {raw_filename}")
print(f"Total albums scraped: {len(df)}")