"""
download_data.py — Fetch oscillator census data from Catagolue.

Uses the CSV endpoint /textcensus/b3s23/C1/xp{n} for clean, complete data,
and the statistics page for aggregate counts.

Output (in data/):
  statistics.json          — aggregate stats from catagolue.hatsya.com/statistics
  census_xp{n}.json        — list of {apgcode, occurrences, period} per period
  summary.json             — one row per period: n_objects, total_occurrences, top3

Usage:
    python download_data.py [--periods 2-30] [--delay 0.5]
"""

import argparse
import csv
import io
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://catagolue.hatsya.com"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def fetch(url, delay=0.5, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30,
                             headers={"User-Agent": "periodic-study-bot/1.0 (research)"})
            r.raise_for_status()
            time.sleep(delay)
            return r.text
        except Exception as e:
            print(f"    [warn] attempt {attempt+1}/{retries}: {e}")
            time.sleep(delay * 3)
    return None


def fetch_census_csv(period, delay=0.5):
    """Fetch CSV census for a given period. Returns list of dicts."""
    url = f"{BASE}/textcensus/b3s23/C1/xp{period}"
    text = fetch(url, delay=delay)
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    objects = []
    for row in reader:
        apgcode = row.get("apgcode", "").strip().strip('"')
        occ_str = row.get("occurrences", "0").strip().strip('"').replace(",", "")
        try:
            occ = int(occ_str)
        except ValueError:
            continue
        if apgcode:
            objects.append({
                "apgcode": apgcode,
                "occurrences": occ,
                "period": period,
            })
    return objects


def fetch_statistics(delay=0.5):
    """Fetch and parse the main statistics page."""
    url = f"{BASE}/statistics"
    print(f"Fetching {url} …")
    html = fetch(url, delay=delay)
    if not html:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    stats = {}

    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if "soups" in text and "objects" in text and "distinct" in text:
            stats["summary"] = text
            nums = [int(n.replace("\u2009","").replace(",",""))
                    for n in re.findall(r"[\d\u2009]{3,}", text)
                    if n.replace("\u2009","").replace(",","").isdigit()]
            if len(nums) >= 3:
                stats["total_soups"]          = nums[0]
                stats["total_objects"]        = nums[1]
                stats["distinct_object_types"] = nums[2]
            break

    # Find which periods have census links
    period_links = {}
    for a in soup.find_all("a", href=True):
        m = re.search(r"/census/b3s23/C1/xp(\d+)", a["href"])
        if m:
            period_links[int(m.group(1))] = BASE + a["href"]
    stats["period_census_links"] = period_links

    print(f"  {stats.get('summary','')[:120]}")
    print(f"  Census links found for periods: {sorted(period_links)}")
    return stats


def parse_periods(s):
    periods = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            periods.update(range(int(lo), int(hi)+1))
        else:
            periods.add(int(part))
    return sorted(periods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", default="2-30")
    ap.add_argument("--delay",   type=float, default=0.5)
    ap.add_argument("--out",     default=DATA_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    periods = parse_periods(args.periods)
    print(f"Fetching periods: {periods}\n")

    # Statistics
    stats = fetch_statistics(args.delay)
    with open(os.path.join(args.out, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics.json\n")

    summary = []

    for period in periods:
        url = f"{BASE}/textcensus/b3s23/C1/xp{period}"
        print(f"Period {period:3d}  {url}")
        objects = fetch_census_csv(period, args.delay)

        if not objects:
            print(f"  (no objects found)")
            summary.append({"period": period, "n_objects": 0, "total_occurrences": 0})
            continue

        total = sum(o["occurrences"] for o in objects)
        top3  = [o["apgcode"] for o in objects[:3]]
        print(f"  {len(objects):5d} objects   total occurrences: {total:,}   top: {top3}")
        summary.append({
            "period":           period,
            "n_objects":        len(objects),
            "total_occurrences": total,
            "top3":             top3,
        })

        out_path = os.path.join(args.out, f"census_xp{period}.json")
        with open(out_path, "w") as f:
            json.dump(objects, f, indent=2)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary.json")

    # Print a quick table
    print("\n--- Summary ---")
    print(f"{'Period':>8}  {'Objects':>8}  {'Total occurrences':>22}")
    for row in summary:
        print(f"  xp{row['period']:<5}  {row['n_objects']:>8}  {row['total_occurrences']:>22,}")
    print("Done.")


if __name__ == "__main__":
    main()
