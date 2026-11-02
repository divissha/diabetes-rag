"""
Pulls PubMed abstracts + metadata on Type 2 diabetes drug therapy.
Output: data/raw/diabetes_papers.json
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUTPUT_PATH = "data/raw/diabetes_papers.json"

# drug-class tag -> PubMed query
SEARCH_QUERIES = {
    "metformin": "type 2 diabetes AND metformin treatment",
    "glp1": "type 2 diabetes AND GLP-1 receptor agonist",
    "sglt2": "type 2 diabetes AND SGLT2 inhibitor",
    "insulin": "type 2 diabetes AND insulin therapy guidelines",
}
PAPERS_PER_QUERY = 250
FETCH_BATCH = 100


def search_pubmed(query, max_results):
    params = {"db": "pubmed", "term": query, "retmax": max_results,
              "retmode": "json", "sort": "relevance"}
    resp = requests.get(f"{BASE_URL}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_xml(pubmed_ids):
    params = {"db": "pubmed", "id": ",".join(pubmed_ids),
              "rettype": "abstract", "retmode": "xml"}
    resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def get_year(article):
    year = article.findtext(".//JournalIssue/PubDate/Year")
    if not year:
        medline = article.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
        m = re.search(r"\d{4}", medline)
        year = m.group(0) if m else "0"
    return int(year)


def parse_records(xml_text):
    root = ET.fromstring(xml_text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        abstract = " ".join(
            "".join(p.itertext()) for p in article.findall(".//AbstractText")
        ).strip()
        if not abstract:
            continue

        pub_types = [p.text for p in article.findall(".//PublicationType") if p.text]
        records.append({
            "pmid": article.findtext(".//PMID"),
            "title": "".join(article.find(".//ArticleTitle").itertext()),
            "abstract": abstract,
            "year": get_year(article),
            "journal": article.findtext(".//Journal/Title") or "",
            "is_rct": any("Randomized Controlled Trial" in t for t in pub_types),
            "is_meta_analysis": any("Meta-Analysis" in t for t in pub_types),
            "is_review": any("Review" in t for t in pub_types),
            "is_guideline": any("Guideline" in t for t in pub_types),
        })
    return records


def main():
    os.makedirs("data/raw", exist_ok=True)
    papers = {}  # pmid -> record (dedupes across queries)

    for drug_class, query in SEARCH_QUERIES.items():
        print(f"Searching [{drug_class}]: {query}")
        ids = search_pubmed(query, PAPERS_PER_QUERY)
        print(f"  Found {len(ids)} IDs")

        for i in range(0, len(ids), FETCH_BATCH):
            batch = ids[i:i + FETCH_BATCH]
            for rec in parse_records(fetch_xml(batch)):
                if rec["pmid"] not in papers:
                    papers[rec["pmid"]] = rec
                papers[rec["pmid"]][f"in_{drug_class}"] = True
            time.sleep(0.5)

    # make sure every flag exists on every record
    for rec in papers.values():
        for drug_class in SEARCH_QUERIES:
            rec.setdefault(f"in_{drug_class}", False)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(list(papers.values()), f, indent=2)
    print(f"\nSaved {len(papers)} unique papers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
