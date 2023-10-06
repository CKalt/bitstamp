import requests
import json
import os

def get_all_bitstamp_pairs():
    url = "https://www.bitstamp.net/api/v2/ticker/all/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Ensure the 'data' directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Save JSON data to a file
        with open("data/bitstamp-all-pairs-ticker.json", "w") as f:
            json.dump(data, f, indent=4)
        
        print("Data saved to data/bitstamp-all-pairs-ticker.json")
        
    except requests.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_all_bitstamp_pairs()
