"""
Pulls PubMed abstracts related to Type 2 diabetes treatment
and saves them as JSON files in data/raw/.
"""

import requests
import time
import json
import os

print("Script started")

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUTPUT_DIR = "data/raw"

# Tune this to your sub-topics — each query pulls a separate batch
SEARCH_QUERIES = [
    "type 2 diabetes AND metformin treatment",
    "type 2 diabetes AND GLP-1 receptor agonist",
    "type 2 diabetes AND SGLT2 inhibitor",
    "type 2 diabetes AND insulin therapy guidelines",
]

PAPERS_PER_QUERY = 100  # keep small at first to test the pipeline


def search_pubmed(query, max_results):
    """Get a list of PubMed IDs matching the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_abstracts(pubmed_ids):
    """Fetch title + abstract for a batch of PubMed IDs."""
    if not pubmed_ids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params)
    resp.raise_for_status()
    return resp.text  # raw XML — we'll parse it in the next function


def parse_xml_to_records(xml_text):
    """Extract title, abstract, and PMID from PubMed XML."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle") or ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(part.text or "" for part in abstract_parts)

        if abstract.strip():  # skip papers with no abstract
            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
            })

    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_records = []

    for query in SEARCH_QUERIES:
        print(f"Searching: {query}")
        ids = search_pubmed(query, PAPERS_PER_QUERY)
        print(f"  Found {len(ids)} papers")

        xml_text = fetch_abstracts(ids)
        records = parse_xml_to_records(xml_text)
        print(f"  Parsed {len(records)} with usable abstracts")

        all_records.extend(records)
        time.sleep(0.5)  # be polite to NCBI's servers

    # de-duplicate by PMID in case queries overlap
    seen = set()
    unique_records = []
    for r in all_records:
        if r["pmid"] not in seen:
            seen.add(r["pmid"])
            unique_records.append(r)

    output_path = os.path.join(OUTPUT_DIR, "diabetes_papers.json")
    with open(output_path, "w") as f:
        json.dump(unique_records, f, indent=2)

    print(f"\nSaved {len(unique_records)} unique papers to {output_path}")


if __name__ == "__main__":
    main()