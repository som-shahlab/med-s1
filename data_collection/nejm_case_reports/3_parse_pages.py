"""
Usage:

    python 3_parse_pages.py

Purpose:

    - Parse the frontmatter, bodymatter, and backmatter of each downloaded NEJM Case Report HTML page.
    - Save the parsed data to a JSON file.

"""
import json
import os
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from datetime import datetime
import multiprocessing
from functools import partial
from tqdm import tqdm
from utils import load_collected_urls_df
from markdownify import markdownify as md

def parse_frontmatter(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract title
    title_tag = soup.find("h1", property="name")
    title = title_tag.get_text(strip=True) if title_tag else None
    
    # Extract authors: join givenName and familyName (include honorificSuffix if present)
    authors = []
    for author_tag in soup.find_all("span", property="author"):
        given = author_tag.find("span", property="givenName")
        family = author_tag.find("span", property="familyName")
        suffix = author_tag.find("span", property="honorificSuffix")
        name = ""
        if given:
            name += given.get_text(strip=True) + " "
        if family:
            name += family.get_text(strip=True)
        if suffix:
            name += ", " + suffix.get_text(strip=True)
        authors.append(name.strip())
    
    # Extract published date and convert to YYYY-MM-DD format
    date_tag = soup.find("span", property="datePublished")
    published_date = None
    if date_tag:
        try:
            date_obj = datetime.strptime(date_tag.get_text(strip=True), "%B %d, %Y")
            published_date = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            published_date = date_tag.get_text(strip=True)
    
    # Extract Volume and No using the core-enumeration section
    volume_tag = soup.select_one("div.core-self-citation span[property='volumeNumber']")
    volume = volume_tag.get_text(strip=True) if volume_tag else None
    
    issue_tag = soup.select_one("div.core-enumeration span[property='issueNumber']")
    issue = issue_tag.get_text(strip=True) if issue_tag else None
    
    return {
        "authors": authors,
        "published_date": published_date,
        "volume": volume,
        "issue": issue,
        "title": title
    }

def parse_bodymatter(html: str) -> Dict[str, Any]:
    """
    Processes the given HTML string by:
      1. Removing all <div> elements with class "component-video"
      2. Removing all <div> elements with class "figure-wrap"
      3. Removing all <div> elements with class "core-first-page-image" (these are .png of PDFs)
      4. Extracting all <section> elements whose <h2> contains "Presentation of Case"
         into a dictionary (key: the h2 text, value: the section's text content),
         and removing those sections from the DOM.
      5. Extracting all <section> elements whose <h2> contains "Diagnosis"
         into a dictionary (key: the h2 text, value: the section's text content),
         and removing those sections from the DOM.
      6. Converting the remaining HTML to markdown.
    
    Returns:
        intro_sections (dict): Extracted sections with presentation-of-case-related <h2>.
        diagnosis_sections (dict): Extracted sections with diagnosis-related <h2>.
        markdown (str): Markdown conversion of the remaining HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. Remove all divs with class "component-video"
    for div in soup.find_all("div", class_="component-video"):
        div.decompose()
    
    # 2. Remove all divs with class "figure-wrap"
    for div in soup.find_all("div", class_="figure-wrap"):
        div.decompose()
    
    # 3. Remove all divs with class "core-first-page-image" (these are .png of PDFs)
    for div in soup.find_all("div", class_="core-first-page-image"):
        div.decompose()
    
    # 4. Extract sections with <h2> of "Presentation of Case" and remove them from the DOM
    intro_sections = {}
    for section in soup.find_all("section"):
        h2 = section.find("h2")
        if h2 and "presentation of case" in h2.get_text().lower():
            # Clone the section and remove h2 to avoid including it in the text
            section_clone = BeautifulSoup(str(section), 'html.parser')
            h2_tag = section_clone.find("h2")
            if h2_tag:
                h2_tag.decompose()
            intro_sections["presentation_of_case"] = section_clone.get_text(separator="\n", strip=True)
            section.decompose()

    # 5. Extract sections with <h2> containing "Diagnosis"
    diagnosis_sections = {}
    for section in soup.find_all("section"):
        h2 = section.find("h2")
        if (h2 # <h2> found
            and (
                # <h2> contains "Diagnosis" / "Diagnoses"
                "diagnosis" in h2.get_text().lower()
                or "diagnoses" in h2.get_text().lower()
            )
            # Ignore "Differential Diagnosis" sections
            and "differential" not in h2.get_text().lower()
            # Ignore "Discussion Of " sections
            and "discussion of" not in h2.get_text().lower()
        ):
            key = h2.get_text(strip=True)
            # Clone the section and remove h2 to avoid including it in the text
            section_clone = BeautifulSoup(str(section), 'html.parser')
            h2_tag = section_clone.find("h2")
            if h2_tag:
                h2_tag.decompose()
            # Using get_text with newlines for readability
            diagnosis_sections[key] = section_clone.get_text(separator="\n", strip=True)
            section.decompose()
    
    # 5. Convert the remaining HTML to markdown
    remaining_html = str(soup)
    markdown = md(remaining_html)
    
    # Set to None if empty
    if markdown == "":
        markdown = None
    if len(diagnosis_sections) == 0:
        diagnosis_sections = None
    
    return {
        "intro_sections" : intro_sections,
        "diagnosis_sections" : diagnosis_sections,
        "markdown" : markdown,
    }

def parse_backmatter(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract the back matter text
    back_matter_tag = soup.find("div", class_="back-matter")
    if back_matter_tag:
        back_matter = back_matter_tag.get_text(strip=True)
    else:
        back_matter = None
    
    return {
        "content": back_matter
    }

def parse_page(path_to_dir: str) -> Dict[str, Any]:
    dir_name = os.path.basename(path_to_dir)
    path_to_frontmatter_html = os.path.join(path_to_dir, "frontmatter.html")
    path_to_bodymatter_html = os.path.join(path_to_dir, "bodymatter.html")
    path_to_backmatter_html = os.path.join(path_to_dir, "backmatter.html")
    
    frontmatter, bodymatter, backmatter = None, None, None
    if os.path.exists(path_to_frontmatter_html):
        with open(path_to_frontmatter_html, "r") as f:
            frontmatter = parse_frontmatter(f.read())
    if os.path.exists(path_to_bodymatter_html):
        with open(path_to_bodymatter_html, "r") as f:
            bodymatter = parse_bodymatter(f.read())
    if os.path.exists(path_to_backmatter_html):
        with open(path_to_backmatter_html, "r") as f:
            backmatter = parse_backmatter(f.read())

    return {
        "id": dir_name,
        "url" : f"https://www.nejm.org/doi/full/10.1056/{dir_name}",
        "frontmatter": frontmatter,
        "bodymatter": bodymatter,
        "backmatter": backmatter
    }
    
def parse_chunk(chunk: List[str], path_to_output_dir: str) -> None:
    """Process a chunk of directory paths."""
    for path_to_dir in chunk:
        dir_name = os.path.basename(path_to_dir)
        path_to_output_json = os.path.join(path_to_output_dir, f"{dir_name}.json")
        
        # Parse HTML
        parsed_data = parse_page(path_to_dir)
        with open(path_to_output_json, "w") as f:
            json.dump(parsed_data, f, indent=2)


# Example usage:
if __name__ == "__main__":
    n_workers: int = 10
    
    path_to_urls_dir: str = "outputs/1_collect_urls/"
    path_to_input_dir: str = "outputs/2_scrape_pages/"
    path_to_output_dir: str = "outputs/3_parse_pages/"
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Load collected URLs + scraped HTML
    paths_to_dirs: List[str] = sorted([ os.path.join(path_to_input_dir, d) for d in os.listdir(path_to_input_dir) if os.path.isdir(os.path.join(path_to_input_dir, d)) ])
    print(f"Found {len(paths_to_dirs)} directories in {path_to_input_dir}.")

    # Create chunks of directories (10 dirs per chunk)
    chunk_size = 10
    chunks = [paths_to_dirs[i:i + chunk_size] for i in range(0, len(paths_to_dirs), chunk_size)]
    
    # Create a partial function with the output directory
    process_chunk_with_output = partial(parse_chunk, path_to_output_dir=path_to_output_dir)
    
    # Process chunks in parallel with progress bar
    with multiprocessing.Pool(processes=n_workers) as pool:
        _ = list(tqdm(pool.imap(process_chunk_with_output, chunks), total=len(chunks), desc="Processing pages"))
    
    print(f"Finished processing {len(paths_to_dirs)} directories. Results saved to {path_to_output_dir}")
    
    # Stats
    parsed_jsons = [ os.path.join(path_to_output_dir, f) for f in os.listdir(path_to_output_dir) if f.endswith(".json") ]
    jsons = [ json.load(open(f, "r")) for f in parsed_jsons ]
    print(f"Found {len(parsed_jsons)} parsed JSONs in {path_to_output_dir}.")
    n_frontmatter = sum([ 1 for j in jsons if j["frontmatter"] is not None and j['frontmatter'].get('authors') is not None ])
    n_bodymatter = sum([ 1 for j in jsons if j["bodymatter"] is not None and j['bodymatter'].get('markdown') is not None ])
    n_backmatter = sum([ 1 for j in jsons if j["backmatter"] is not None and j['backmatter'].get('content') is not None ])
    print(f"# of JSONS with frontmatter: {n_frontmatter}")
    print(f"# of JSONS with bodymatter: {n_bodymatter}")
    print(f"# of JSONS with backmatter: {n_backmatter}")
