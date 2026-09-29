from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


TRAFFIC_PAGE = "https://paradipport.gov.in/traffic/"


response = requests.get(
    TRAFFIC_PAGE,
    timeout=30,
    headers={
        "User-Agent": "NEREUS/0.1",
    },
)

response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

print("Paradip traffic reports found:")
print("=" * 70)

for link in soup.find_all("a", href=True):
    text = link.get_text(" ", strip=True)
    href = urljoin(TRAFFIC_PAGE, link["href"])

    if "Traffic Report" in text or "traffic report" in text:
        print(f"TEXT: {text}")
        print(f"URL:  {href}")
        print()