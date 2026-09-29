import requests
from bs4 import BeautifulSoup

URL = "https://www.balticdryindex.com/bdi-chart-historical-data/"

headers = {
    "User-Agent": "NEREUS/0.1"
}

print("Fetching:", URL)

r = requests.get(URL, headers=headers, timeout=30)

print("HTTP:", r.status_code)
print("Content-Type:", r.headers.get("content-type"))
print("Bytes:", len(r.content))

r.raise_for_status()

text = r.text

print("\n--- Source indicators ---")
for term in ["Stooq", "Baltic Exchange", "BCI", "BPI", "BSI", "BHSI", "BDI"]:
    print(f"{term}: {'YES' if term.lower() in text.lower() else 'NO'}")

print("\n--- Script sources ---")

soup = BeautifulSoup(text, "html.parser")

for script in soup.find_all("script"):
    src = script.get("src")
    if src:
        print(src)

print("\nDiagnostic complete.")
