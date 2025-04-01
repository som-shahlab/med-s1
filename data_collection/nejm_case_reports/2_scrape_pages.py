"""
Usage:

    python 2_scrape_pages.py

Purpose:

    - Scrape the frontmatter, bodymatter, and backmatter of each NEJM Case Report page.
    - Save the HTML content to files.
"""

import os
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import concurrent.futures
from tqdm import tqdm
import traceback
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import load_collected_urls_df

def init_driver(port: int):
    chrome_debugger_address = f"127.0.0.1:{port}"  # Ensure this matches your Chrome debug session
    options = Options()
    options.debugger_address = chrome_debugger_address  # Connect to running Chrome
    driver = webdriver.Chrome(options=options)
    return driver

def url_to_page_id(url: str) -> str:
    return url.split("/")[-1]

def process_url(driver: webdriver.Chrome, url: str, path_to_output_dir: str):
    # Parse the DOI id from the URL (the last part)
    page_id: int = url_to_page_id(url)
    
    url = url.replace("https://www.nejm.org", "https://www-nejm-org.laneproxy.stanford.edu")
    
    try:
        driver.get(url)
        # Wait for bodymatter element to load
        # WebDriverWait makes Selenium wait for a certain condition before proceeding
        # It helps handle dynamic content that may take time to load
        # In this case, it waits up to 5 seconds for the "bodymatter" element to appear
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "bodymatter"))
        )
    
        # Extract the inner HTML from the specified divs.
        frontmatter_html = ""
        try:
            frontmatter_element = driver.find_element(By.CSS_SELECTOR, "header[data-extent='frontmatter']")
            if frontmatter_element:
                frontmatter_html = frontmatter_element.get_attribute("innerHTML")
        except Exception:
            pass  # Element doesn't exist, keep frontmatter_html as empty string
        body_html = ""
        try:
            body_element = driver.find_element(By.ID, "bodymatter")
            if body_element:
                body_html = body_element.get_attribute("innerHTML")
        except Exception:
            pass  # Element doesn't exist, keep body_html as empty string
        backmatter_html = ""
        try:
            backmatter_element = driver.find_element(By.ID, "backmatter")
            if backmatter_element:
                backmatter_html = backmatter_element.get_attribute("innerHTML")
        except Exception:
            pass  # Element doesn't exist, keep backmatter_html as empty string
    
        # Save the HTML content to files.
        os.makedirs(os.path.join(path_to_output_dir, page_id), exist_ok=True) # ! NOTE: Must be located here b/c we use the existence of the folder to determine if the page has already been scraped.
        with open(os.path.join(path_to_output_dir, page_id, "frontmatter.html"), "w", encoding="utf-8") as f:
            f.write(frontmatter_html)
        with open(os.path.join(path_to_output_dir, page_id, "bodymatter.html"), "w", encoding="utf-8") as f:
            f.write(body_html)
        with open(os.path.join(path_to_output_dir, page_id, "backmatter.html"), "w", encoding="utf-8") as f:
            f.write(backmatter_html)
    except Exception as e:
        print(f"Error processing {url}: {e}")

def process_url_chunk(urls: List[str], path_to_output_dir: str, port: int):
    """Process a chunk of URLs in a single process"""
    driver = None
    try:
        driver = init_driver(port)
        print(f"Processing URLs on port {port}")
        for url in urls:
            page_id: int = url_to_page_id(url)
            if os.path.exists(os.path.join(path_to_output_dir, page_id)):
                print(f"Skipping {url} because it already exists.")
                continue
            try:
                process_url(driver, url, path_to_output_dir)
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {url}: {e}")
    finally:
        print(f"Quitting driver on port {port}")
        if driver:
            driver.quit()

if __name__ == "__main__":
    # Create and start a thread for each URL.
    path_to_input_dir: str = "outputs/1_collect_urls/"
    path_to_output_dir: str = "outputs/2_scrape_pages/"
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Load URLs from CSVs
    df = load_collected_urls_df(path_to_input_dir)
    urls = sorted(df["page_url"].tolist())
    print(f"Loaded {len(urls)} URLs from {len(df)} CSV files")
    
    # Sanity checks.
    # assert len(urls) == 7115, f"Expected 7115 NEJM Case Reports URLs, got {len(urls)}"

    n_workers = 5
    
    # Init unique chrome instance for each port.
    ports = [9222 + i for i in range(n_workers)]
    
    # Split URLs into chunks of size (len(urls) / n_workers)
    url_chunks = [urls[i:i + len(urls) // n_workers] for i in range(0, len(urls), len(urls) // n_workers)]
    if n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit chunks to process pool
            futures = [executor.submit(process_url_chunk, chunk, path_to_output_dir, ports[idx]) for idx, chunk in enumerate(url_chunks)]
            
            # Process results as they complete with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(url_chunks), 
                            desc="Processing URLs"):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error occurred: {e}")
    else:
        for chunk in tqdm(url_chunks, desc="Processing URLs"):
            process_url_chunk(chunk, path_to_output_dir, ports[0])
        
    print("All URLs have been processed.")