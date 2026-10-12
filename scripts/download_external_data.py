#!/usr/bin/env python3
"""Download licensed external data for PRC 2026.

The core scope downloads the 2022 PRC flight list and ERA5 point weather for
the 11 competition airports. The trajectories scope downloads the complete,
large 2022 daily trajectory archive as individually resumable files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "external"
FOUR_TU_LANDING_URL = (
    "https://data.4tu.nl/datasets/"
    "8cb8484b-dbe7-4750-8b87-a5b1dbc621b4/2"
)
FOUR_TU_FILE_BASE = "https://data.4tu.nl"
EXPECTED_TRAJECTORY_FILES = 365
EXPECTED_TRAJECTORY_BYTES = 306_373_396_928
FLIGHT_LIST = {
    "name": "flight_list.csv",
    "bytes": 101_091_663,
    "md5": "2da9db49afcae255eab0828f1e2f8575",
    "url": (
        "https://data.4tu.nl/file/"
        "8cb8484b-dbe7-4750-8b87-a5b1dbc621b4/"
        "2bfbef1a-a75b-4045-a46e-3325bf363c27"
    ),
}
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_START = "2024-01-01"
WEATHER_END = "2025-12-31"
WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_10m",
    "wind_direction_10m",
)
WEATHER_CSV_COLUMNS = (
    "time",
    "temperature_2m (°C)",
    "relative_humidity_2m (%)",
    "dew_point_2m (°C)",
    "precipitation (mm)",
    "rain (mm)",
    "snowfall (cm)",
    "pressure_msl (hPa)",
    "cloud_cover (%)",
    "cloud_cover_low (%)",
    "wind_speed_10m (m/s)",
    "wind_direction_10m (°)",
)
EXPECTED_WEATHER_ROWS = 17_544
EXPECTED_WEATHER_FIRST_HOUR = datetime(2024, 1, 1)
EXPECTED_WEATHER_LAST_HOUR = datetime(2025, 12, 31, 23)
AIRPORTS = {
    "EDDF": (50.026, 8.543),
    "EDDM": (48.354, 11.786),
    "EGLL": (51.478, -0.461),
    "EHAM": (52.309, 4.764),
    "LEBL": (41.297, 2.078),
    "LEMD": (40.494, -3.567),
    "LFPG": (49.013, 2.550),
    "LIRF": (41.804, 12.251),
    "LTAI": (36.899, 30.800),
    "LTFM": (41.275, 28.752),
    "LSZH": (47.465, 8.549),
}
WEATHER_SNAPSHOTS = {
    "EDDF": (1_150_735, "8778c1286c20b7e7616ad068c3ebad48968978cda57b3724d48524be138ec513"),
    "EDDM": (1_149_171, "ad5a78e554da7abb3d8a498a4ff8dc586565fe7ef8f0e9aa915f7de0d6066470"),
    "EGLL": (1_154_424, "6a86b696266cb09c0f64f6f4ecb0753330b04d57de0b7b94dd2196447757c864"),
    "EHAM": (1_152_972, "cae2a35552e72b137abcfc33c90de65d3757feda7d83103b63713853ec395707"),
    "LEBL": (1_153_620, "6052e2e31e564d22c51d25ed5daa5400e121179595133e5f3cd74489ef47be9a"),
    "LEMD": (1_137_718, "6bf5aefe4fad05d65b5bd246ff34c66f7ab24cb577265e86882f77e4dde17397"),
    "LFPG": (1_153_866, "8f6aa4cb409d4b17f028a16f5d4f1339f70e5d72153e3f753ea6e09b4888f88c"),
    "LIRF": (1_146_704, "3e85d8a4e77c16884d0b55b3a2a7a61ea83dba2e810a4d840c6b5665eeff0b89"),
    "LTAI": (1_144_704, "ba43e72546881599af7cf9bbe9fce6727248fe30e9d393258cb039b5fe3f12df"),
    "LTFM": (1_142_456, "ef388abf8dcf04d81a855592bd0b5332b1ce1019f5d09f519ef5ebb3a94ef186"),
    "LSZH": (1_148_421, "aac45d8b5cf0c1c1ffe983af2fa6b6fa61e65a2eccce9cca386b761cdb26e21d"),
}


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    bytes: int | None = None
    md5: str | None = None
    sha256: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(path: Path, remote: RemoteFile) -> bool:
    if not path.is_file():
        return False
    if remote.bytes is not None and path.stat().st_size != remote.bytes:
        return False
    if remote.md5 is not None and digest(path, "md5") != remote.md5:
        return False
    if remote.sha256 is not None and digest(path, "sha256") != remote.sha256:
        return False
    if remote.bytes is None and remote.md5 is None and remote.sha256 is None:
        return False
    return True


def download_resumable(remote: RemoteFile, destination: Path, retries: int = 8) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verify(destination, remote):
        print(f"verified existing {destination} ({destination.stat().st_size:,} bytes)", flush=True)
        return destination

    part = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "prc-data-challenge-2026/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(remote.url, headers=headers)
            print(
                f"download {remote.name}: attempt={attempt} resume_byte={offset:,}",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", response.getcode())
                mode = "ab" if offset and status == 206 else "wb"
                with part.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)

            if remote.bytes is not None and part.stat().st_size != remote.bytes:
                actual_bytes = part.stat().st_size
                if actual_bytes > remote.bytes:
                    part.unlink()
                raise IOError(
                    f"size mismatch for {remote.name}: "
                    f"{actual_bytes:,} != {remote.bytes:,}"
                )
            if remote.md5 is not None and digest(part, "md5") != remote.md5:
                part.unlink()
                raise IOError(f"MD5 mismatch for {remote.name}")
            if remote.sha256 is not None and digest(part, "sha256") != remote.sha256:
                part.unlink()
                raise IOError(f"SHA-256 mismatch for {remote.name}")
            part.replace(destination)
            print(f"complete {destination} ({destination.stat().st_size:,} bytes)", flush=True)
            return destination
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            print(f"retry {remote.name}: {error}", file=sys.stderr, flush=True)
            if attempt == retries:
                raise
            time.sleep(min(60, 2**attempt))
    raise AssertionError("unreachable")


def weather_url(latitude: float, longitude: float) -> str:
    parameters = {
        "latitude": f"{latitude:.3f}",
        "longitude": f"{longitude:.3f}",
        "start_date": WEATHER_START,
        "end_date": WEATHER_END,
        "hourly": ",".join(WEATHER_VARIABLES),
        "models": "era5",
        "timezone": "GMT",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "format": "csv",
    }
    return WEATHER_API_URL + "?" + urllib.parse.urlencode(parameters, safe=",")


def validate_weather_csv(path: Path) -> bool:
    """Verify the expected two-year, gap-free Open-Meteo CSV structure."""
    if not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            metadata_header = next(reader)
            metadata_values = next(reader)
            blank_line = next(reader)
            weather_header = next(reader)
            if metadata_header != [
                "latitude",
                "longitude",
                "elevation",
                "utc_offset_seconds",
                "timezone",
                "timezone_abbreviation",
            ]:
                return False
            if len(metadata_values) != len(metadata_header) or metadata_values[3:] != [
                "0",
                "GMT",
                "GMT",
            ]:
                return False
            if blank_line or tuple(weather_header) != WEATHER_CSV_COLUMNS:
                return False

            count = 0
            last_timestamp = None
            for row in reader:
                if len(row) != len(WEATHER_CSV_COLUMNS):
                    return False
                timestamp = datetime.fromisoformat(row[0])
                expected = EXPECTED_WEATHER_FIRST_HOUR + timedelta(hours=count)
                if timestamp != expected:
                    return False
                last_timestamp = timestamp
                count += 1
    except (OSError, UnicodeDecodeError, ValueError, StopIteration):
        return False
    return (
        count == EXPECTED_WEATHER_ROWS
        and last_timestamp == EXPECTED_WEATHER_LAST_HOUR
    )


def download_core(
    data_dir: Path,
    allow_open_meteo_free_api: bool = False,
) -> None:
    flight_remote = RemoteFile(
        name=str(FLIGHT_LIST["name"]),
        url=str(FLIGHT_LIST["url"]),
        bytes=int(FLIGHT_LIST["bytes"]),
        md5=str(FLIGHT_LIST["md5"]),
    )
    flight_path = data_dir / "prc2024" / flight_remote.name
    download_resumable(flight_remote, flight_path)

    weather_dir = data_dir / "weather" / "era5"
    weather_dir.mkdir(parents=True, exist_ok=True)
    for airport, (latitude, longitude) in AIRPORTS.items():
        name = f"{airport}_era5_{WEATHER_START}_{WEATHER_END}.csv"
        expected_bytes, expected_sha256 = WEATHER_SNAPSHOTS[airport]
        remote = RemoteFile(
            name=name,
            url=weather_url(latitude, longitude),
            bytes=expected_bytes,
            sha256=expected_sha256,
        )
        destination = weather_dir / name
        if verify(destination, remote):
            print(
                f"verified existing {destination} "
                f"({destination.stat().st_size:,} bytes)",
                flush=True,
            )
            path = destination
        else:
            if not allow_open_meteo_free_api:
                raise PermissionError(
                    "Open-Meteo weather retrieval is disabled because the free "
                    "API terms are unresolved for prize use. Use an approved "
                    "retrieval route, or pass --allow-open-meteo-free-api only "
                    "after you document a valid non-commercial basis."
                )
            path = download_resumable(remote, destination)
        if not validate_weather_csv(path):
            raise ValueError(f"Weather completeness check failed for {path}")

    records = []
    for path in [flight_path, *sorted(weather_dir.glob("*.csv"))]:
        if path == flight_path:
            source_url = flight_remote.url
            source = "PRC 2024 open dataset v2"
        else:
            airport = path.name.split("_", 1)[0]
            latitude, longitude = AIRPORTS[airport]
            source_url = weather_url(latitude, longitude)
            source = "Open-Meteo ERA5"
        records.append(
            {
                "file": str(path.relative_to(data_dir)),
                "bytes": path.stat().st_size,
                "sha256": digest(path, "sha256"),
                "source": source,
                "source_url": source_url,
                "license": "CC BY 4.0",
                "verified_at_utc": utc_now(),
            }
        )
    manifest_path = data_dir / "core_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {manifest_path}", flush=True)


def parse_trajectory_manifest(html: str) -> list[RemoteFile]:
    files = []
    for block in re.findall(r"<li>.*?</li>", html, flags=re.DOTALL):
        name_match = re.search(r">(2022-\d{2}-\d{2}\.parquet)</a>", block)
        if name_match is None:
            continue
        size_match = re.search(r'<span class="size">([\d,]+) bytes</span>', block)
        md5_match = re.search(
            r'<span class="md5">MD5:<code>([a-f0-9]{32})</code></span>', block
        )
        path_match = re.search(
            r'<a class="corporate-identity" href="([^"]+)">', block
        )
        if size_match is None or md5_match is None or path_match is None:
            raise ValueError(f"Incomplete metadata for {name_match.group(1)}")
        files.append(
            RemoteFile(
                name=name_match.group(1),
                url=urllib.parse.urljoin(FOUR_TU_FILE_BASE, path_match.group(1)),
                bytes=int(size_match.group(1).replace(",", "")),
                md5=md5_match.group(1),
            )
        )
    if len(files) != EXPECTED_TRAJECTORY_FILES:
        raise ValueError(
            f"Expected {EXPECTED_TRAJECTORY_FILES} daily trajectory files, "
            f"found {len(files)}"
        )
    total_bytes = sum(item.bytes or 0 for item in files)
    if total_bytes != EXPECTED_TRAJECTORY_BYTES:
        raise ValueError(
            f"Trajectory byte total changed: {total_bytes:,} != "
            f"{EXPECTED_TRAJECTORY_BYTES:,}"
        )
    priority = {
        "2022-01-25.parquet": 0,
        "2022-07-01.parquet": 1,
        "2022-01-01.parquet": 2,
    }
    return sorted(files, key=lambda item: (priority.get(item.name, 3), item.name))


def download_trajectories(data_dir: Path, workers: int) -> None:
    request = urllib.request.Request(
        FOUR_TU_LANDING_URL, headers={"User-Agent": "prc-data-challenge-2026/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        html = response.read().decode("utf-8")
    files = parse_trajectory_manifest(html)

    trajectory_dir = data_dir / "prc2024" / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    remaining_bytes = 0
    for remote in files:
        destination = trajectory_dir / remote.name
        part = destination.with_name(destination.name + ".part")
        present_bytes = destination.stat().st_size if destination.exists() else 0
        if not present_bytes and part.exists():
            present_bytes = part.stat().st_size
        remaining_bytes += max(0, (remote.bytes or 0) - present_bytes)
    free_bytes = shutil.disk_usage(trajectory_dir).free
    reserve_bytes = 20 * 1024**3
    if free_bytes < remaining_bytes + reserve_bytes:
        raise OSError(
            f"Insufficient disk space: need {remaining_bytes + reserve_bytes:,} "
            f"bytes including reserve, have {free_bytes:,}"
        )
    manifest_path = data_dir / "prc2024" / "trajectory_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": FOUR_TU_LANDING_URL,
                "generated_at_utc": utc_now(),
                "files": [asdict(item) for item in files],
                "total_bytes": sum(item.bytes or 0 for item in files),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"trajectory manifest: {len(files)} files, "
        f"{sum(item.bytes or 0 for item in files):,} bytes",
        flush=True,
    )

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_resumable, remote, trajectory_dir / remote.name): remote
            for remote in files
        }
        for future in as_completed(futures):
            remote = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - collect all failed files
                failures.append((remote.name, repr(error)))
                print(f"failed {remote.name}: {error!r}", file=sys.stderr, flush=True)
    if failures:
        failure_path = data_dir / "prc2024" / "trajectory_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} trajectory downloads failed; see {failure_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=("core", "trajectories", "all"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("PRC_EXTERNAL_DATA_DIR", DEFAULT_DATA_DIR)),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("PRC_EXTERNAL_WORKERS", "2")),
    )
    parser.add_argument(
        "--allow-open-meteo-free-api",
        action="store_true",
        help=(
            "allow weather retrieval only after documenting a valid "
            "non-commercial basis under the Open-Meteo free API terms"
        ),
    )
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.workers > 4:
        parser.error("--workers must be between 1 and 4")

    if arguments.scope in {"core", "all"}:
        download_core(arguments.data_dir, arguments.allow_open_meteo_free_api)
    if arguments.scope in {"trajectories", "all"}:
        download_trajectories(arguments.data_dir, arguments.workers)


if __name__ == "__main__":
    main()
