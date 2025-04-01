"""
Usage:
    bash 1_collect_urls.sh
    python 1_collect_urls.py

Purpose:

    - Collect the URLs of all NEJM Case Reports.
    - Save the URLs to a CSV file.

"""
import os
import concurrent.futures
import traceback
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from tqdm import tqdm

chrome_debugger_address = "127.0.0.1:9222"  # Ensure this matches your Chrome debug session
options = Options()
options.debugger_address = chrome_debugger_address  # Connect to running Chrome
driver = webdriver.Chrome(options=options)

# Function to initialize a Selenium Chrome driver attached to your current instance
def init_driver():
    options = Options()
    # Connect to the Chrome instance running on port 9222
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    # Make sure to specify the correct path to your chromedriver if it's not in your PATH
    driver = webdriver.Chrome(options=options)
    return driver

# Function to scrape one page given its page number
def scrape_page(page_num: int, path_to_output_dir: str):
    path_to_output_csv = os.path.join(path_to_output_dir, f"page_{page_num}.csv")
    if os.path.exists(path_to_output_csv):
        try:
            _ = pd.read_csv(path_to_output_csv)
            print(f"Skipping page {page_num} because it already exists")
            return
        except Exception as e:
            # Empty file, so we can proceed.
            pass
    
    url = f"https://www.nejm.org/browse/nejm-article-type/case-records-of-the-massachusetts-general-hospital?startPage={page_num}"
    driver = init_driver()
    driver.get(url)
    
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    results = []
    # Find all links with the target class
    for link in soup.find_all('a', class_="issue-item_title-link animation-underline"):
        href = link.get('href')
        title = link.get_text(strip=True)
        results.append({'page_url': f"https://www.nejm.org{href}", 'source_url': url, 'title': title})
    
    # Save results to CSV
    pd.DataFrame(results).to_csv(path_to_output_csv, index=False)
    driver.quit()

def main():
    n_workers = 1
    total_pages = 356  # iterate pages 1 to 356

    # Output directory
    path_to_output_dir = "outputs/1_collect_urls/"
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    if n_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(scrape_page, page, path_to_output_dir) for page in range(1, total_pages + 1)]
            
            # Wrap concurrent.futures.as_completed with tqdm
            for future in tqdm(concurrent.futures.as_completed(futures), 
                            total=total_pages, 
                            desc="Scraping pages"):
                try:
                    _ = future.result()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error occurred: {e}")
    else:
        for page in tqdm(range(1, total_pages + 1), desc="Scraping pages"):
            scrape_page(page, path_to_output_dir)

if __name__ == '__main__':
    main()
