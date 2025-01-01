import requests
import csv
from datetime import datetime


def fetch_weather_observations():
    url = "http://www.bom.gov.au/fwo/IDN60801/IDN60801.95695.json"

    try:
        print("Fetching weather observations...")
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Referer": "https://www.bom.gov.au/",
                "Origin": "https://www.bom.gov.au/",
            },
        )
        response.raise_for_status()

        print("Fetch completed, processing response...")
        data = response.json()
        observations = data.get("observations", {}).get("data", [])

        if not observations:
            print("No observations found.")
            return

        # Read existing rows from the CSV if it exists
        existing_rows = []
        try:
            with open(
                "weather_observations.csv", mode="r", newline="", encoding="utf-8"
            ) as file:
                reader = csv.reader(file)
                existing_rows = list(reader)
        except FileNotFoundError:
            print("CSV file does not exist. A new file will be created.")

        # Extract existing timestamps from the CSV (skip header if it exists)
        existing_timestamps = set()
        if len(existing_rows) > 1:
            existing_timestamps = {row[0] for row in existing_rows[1:]}

        # Prepare new rows
        new_rows = []
        for obs in observations:
            timestamp = obs.get("local_date_time_full")
            if timestamp and timestamp not in existing_timestamps:
                new_rows.append(
                    [
                        timestamp,
                        obs.get("air_temp"),
                        obs.get("apparent_t"),
                        obs.get("rel_hum"),
                        obs.get("wind_spd_kmh"),
                        obs.get("rain_trace"),
                    ]
                )

        if not new_rows:
            print("No new observations to add.")
            return

        # Combine existing rows (if any) with new rows, maintaining the order
        if len(existing_rows) > 1:
            header = existing_rows[0]
            data_rows = existing_rows[1:]
            combined_rows = data_rows + new_rows
        else:
            header = [
                "timestamp",
                "temperature",
                "apparent_temperature",
                "humidity",
                "wind_speed",
                "rainfall",
            ]
            combined_rows = new_rows

        # Sort rows by timestamp (oldest to latest)
        combined_rows.sort(key=lambda x: x[0])

        # Write updated rows back to the CSV
        with open(
            "weather_observations.csv", mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(combined_rows)

        print("Weather observations saved to 'weather_observations.csv'.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Run the script
if __name__ == "__main__":
    fetch_weather_observations()
