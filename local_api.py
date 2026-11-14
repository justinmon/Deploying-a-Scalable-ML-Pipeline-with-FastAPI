"""Send a GET and a POST to the locally running coursework API."""

import requests

BASE_URL = "http://127.0.0.1:8000"


def main():
    """Print both response statuses, the greeting, and the salary prediction."""
    response = requests.get(f"{BASE_URL}/", timeout=10)
    print(f"GET Status Code: {response.status_code}")
    response.raise_for_status()
    print(f"Welcome: {response.json()['message']}")

    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = requests.post(f"{BASE_URL}/data/", json=data, timeout=10)
    print(f"POST Status Code: {response.status_code}")
    response.raise_for_status()
    print(f"Prediction: {response.json()['result']}")


if __name__ == "__main__":
    main()
