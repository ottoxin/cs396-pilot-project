#!/usr/bin/env python3
import argparse
import csv
import json
import os
import urllib.request


DEFAULT_CSV_URL = "https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv"


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def download_csv(url: str, csv_path: str, force_download: bool = False) -> None:
    ensure_parent_dir(csv_path)
    if os.path.exists(csv_path) and not force_download:
        print(f"CSV already exists at {csv_path}; skipping download.")
        return

    print(f"Downloading AILuminate CSV from {url} -> {csv_path}")
    urllib.request.urlretrieve(url, csv_path)
    print("Download complete.")


def convert_csv_to_jsonl(csv_path: str, jsonl_path: str, add_prompt_alias: bool = True) -> int:
    ensure_parent_dir(jsonl_path)
    count = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f_in, open(
        jsonl_path, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in CSV: {csv_path}")

        for row in reader:
            if add_prompt_alias and "prompt" not in row and "prompt_text" in row:
                row["prompt"] = row["prompt_text"]
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare AILuminate eval data.")
    parser.add_argument("--url", default=DEFAULT_CSV_URL, help="CSV source URL")
    parser.add_argument("--csv-path", default="data/ailuminate_test.csv", help="Local CSV path")
    parser.add_argument("--jsonl-path", default="data/ailuminate.jsonl", help="Output JSONL path")
    parser.add_argument("--skip-download", action="store_true", help="Use existing local CSV")
    parser.add_argument("--force-download", action="store_true", help="Re-download CSV even if it exists")
    parser.add_argument(
        "--no-prompt-alias",
        action="store_true",
        help="Do not add prompt alias from prompt_text",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_csv(args.url, args.csv_path, force_download=args.force_download)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found at {args.csv_path}")

    print(f"Converting {args.csv_path} -> {args.jsonl_path}")
    count = convert_csv_to_jsonl(
        args.csv_path,
        args.jsonl_path,
        add_prompt_alias=not args.no_prompt_alias,
    )
    print(f"Success. Wrote {count} rows to {args.jsonl_path}")


if __name__ == "__main__":
    main()
