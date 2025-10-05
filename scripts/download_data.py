import os
import time
from pathlib import Path
from typing import Dict
import requests
import yaml


def load_config() -> Dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def download_file(url: str, output_path: Path, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            print(f"Downloading from {url} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, timeout=300)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"Successfully saved to {output_path}")
            return True

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return False


def main():
    config = load_config()
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "koi.csv": config["data"]["koi_url"],
        "toi.csv": config["data"]["toi_url"],
        "k2.csv": config["data"]["k2_url"],
    }

    print("Starting NASA dataset download...")

    for filename, url in datasets.items():
        output_path = raw_dir / filename
        if output_path.exists():
            print(f"{filename} already exists, skipping...")
            continue

        success = download_file(url, output_path)
        if not success:
            print(f"Failed to download {filename}")
            return

        time.sleep(2)

    print("\nAll datasets downloaded successfully!")
    print(f"Files saved to: {raw_dir.absolute()}")


if __name__ == "__main__":
    main()
