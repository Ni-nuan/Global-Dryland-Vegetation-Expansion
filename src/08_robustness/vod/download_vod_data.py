import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


RECORD_ID = "17359730"
DOWNLOAD_FOLDER_NAME = "vod_zenodo_17359730"
MAX_RETRIES = 9999
RETRY_WAIT_SECONDS = 30
CHUNK_SIZE = 1024 * 1024  # 1 MB


def get_script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def fetch_zenodo_files(record_id: str):
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Reading Zenodo record: {api_url}")

    with urllib.request.urlopen(api_url, timeout=60) as response:
        record = json.loads(response.read().decode("utf-8"))

    files = []
    for item in record.get("files", []):
        filename = item.get("key", "")
        if not filename.lower().endswith(".zip"):
            continue

        checksum = item.get("checksum", "")
        md5 = checksum.replace("md5:", "") if checksum.startswith("md5:") else None

        files.append({
            "filename": filename,
            "size": int(item.get("size", 0)),
            "md5": md5,
            "url": f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
        })

    files.sort(key=lambda x: x["filename"])
    return files


def file_md5(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_complete_and_valid(path: Path, expected_size: int, expected_md5: str | None) -> bool:
    if not path.exists():
        return False

    if expected_size and path.stat().st_size != expected_size:
        return False

    if expected_md5:
        print(f"Checking MD5: {path.name}")
        actual_md5 = file_md5(path)
        if actual_md5.lower() != expected_md5.lower():
            print(f"MD5 mismatch: {path.name}")
            print(f"Expected: {expected_md5}")
            print(f"Actual:   {actual_md5}")
            return False

    return True


def format_size(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    mb = num_bytes / (1024 ** 2)
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{mb:.1f} MB"


def download_with_resume(url: str, output_path: Path, expected_size: int):
    existing_size = output_path.stat().st_size if output_path.exists() else 0

    headers = {}
    mode = "wb"

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        print(f"Resuming from {format_size(existing_size)}")

    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)

        # If server ignores Range request, restart cleanly to avoid corrupt file.
        if existing_size > 0 and status != 206:
            print("Server did not accept resume request; restarting this file.")
            existing_size = 0
            mode = "wb"
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request, timeout=120)

        downloaded = existing_size
        start_time = time.time()

        with output_path.open(mode) as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break

                f.write(chunk)
                downloaded += len(chunk)

                if expected_size:
                    percent = downloaded / expected_size * 100
                    elapsed = max(time.time() - start_time, 1)
                    speed = (downloaded - existing_size) / elapsed
                    print(
                        f"\r{output_path.name}: "
                        f"{percent:6.2f}% "
                        f"({format_size(downloaded)} / {format_size(expected_size)}), "
                        f"{format_size(int(speed))}/s",
                        end=""
                    )

        print()


def download_file(file_info: dict, download_dir: Path):
    filename = file_info["filename"]
    url = file_info["url"]
    expected_size = file_info["size"]
    expected_md5 = file_info["md5"]

    output_path = download_dir / filename

    if is_complete_and_valid(output_path, expected_size, expected_md5):
        print(f"Already complete: {filename}")
        return

    print("=" * 80)
    print(f"Downloading: {filename}")
    print(f"Size: {format_size(expected_size)}")
    print(f"URL: {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            download_with_resume(url, output_path, expected_size)

            if is_complete_and_valid(output_path, expected_size, expected_md5):
                print(f"Finished: {filename}")
                return

            print(f"Incomplete or invalid file after download: {filename}")
            print(f"Retrying in {RETRY_WAIT_SECONDS} seconds...")

        except urllib.error.HTTPError as e:
            print(f"\nHTTP error on attempt {attempt}: {e}")
        except urllib.error.URLError as e:
            print(f"\nURL error on attempt {attempt}: {e}")
        except TimeoutError as e:
            print(f"\nTimeout on attempt {attempt}: {e}")
        except Exception as e:
            print(f"\nUnexpected error on attempt {attempt}: {e}")

        time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Failed to download after many retries: {filename}")


def main():
    script_dir = get_script_dir()
    base_dir = script_dir / DOWNLOAD_FOLDER_NAME
    zip_dir = base_dir / "zip"
    zip_dir.mkdir(parents=True, exist_ok=True)

    files = fetch_zenodo_files(RECORD_ID)

    if not files:
        raise RuntimeError("No zip files found in the Zenodo record.")

    print(f"Found {len(files)} zip files.")
    print(f"Download folder: {zip_dir}")

    total_size = sum(f["size"] for f in files)
    print(f"Total size: {format_size(total_size)}")

    for i, file_info in enumerate(files, start=1):
        print(f"\nFile {i}/{len(files)}")
        download_file(file_info, zip_dir)

    print("\nAll downloads completed.")
    print(f"Files saved to: {zip_dir}")


if __name__ == "__main__":
    main()