import requests


DEFAULT_TIMEOUT = 30


def get_json(url: str, params: dict | None = None) -> dict | list:
    response = requests.get(
        url,
        params=params,
        timeout=DEFAULT_TIMEOUT,
        headers={
            "User-Agent": "NEREUS/0.1",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()
    return response.json()