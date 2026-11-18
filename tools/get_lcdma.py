"""Download the LCDMA real-time vintages (Fortin-Gagnon, Leroux, Stevanovic and Surprenant, 2022) from the
authors' page into data/lcdma/. Run this on your own computer: the page is not reachable from every network.

    python tools/get_lcdma.py              # every vintage listed on the page
    python tools/get_lcdma.py --last 36    # only the 36 most recent
"""
import argparse
import os
import re
import urllib.parse
import urllib.request
import zipfile

PAGE = "https://www.stevanovic.uqam.ca/DS_LCMD.html"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "data", "lcdma"))
ap.add_argument("--last", type=int, default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
html = urllib.request.urlopen(PAGE, timeout=60).read().decode("utf-8", "ignore")
links = sorted(set(re.findall(r'href="([^"]*LCDMA[^"]*)"', html, re.I)))
if not links:
    raise SystemExit("No LCDMA links found on the page: its layout may have changed. Download the files by hand "
                     f"from {PAGE} into {a.out}.")
links = links[-2 * a.last:] if a.last else links
for href in links:
    url = urllib.parse.urljoin(PAGE, href)
    dest = os.path.join(a.out, os.path.basename(urllib.parse.urlparse(url).path))
    if not os.path.exists(dest):
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    if dest.lower().endswith(".zip"):
        with zipfile.ZipFile(dest) as z:
            z.extractall(a.out)
print(f"{len(links)} files in {a.out}. Check the first one opens as expected, then set source: lcdma in config.yaml.")
