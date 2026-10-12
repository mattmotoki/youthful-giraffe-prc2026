from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from release.era5_data import MANIFEST_PATH, NOTICE, verify_bundle, verify_file

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = json.loads(MANIFEST_PATH.read_text())


def payload(extra: str | None = None, version: int | None = None) -> pd.DataFrame:
    values = {}
    for item in PRODUCTION["schema"]:
        name, dtype = item["name"], item["dtype"]
        if name == "airport":
            values[name] = pd.Series(["EDDF"], dtype="object")
        elif name == "hour_utc":
            values[name] = pd.to_datetime(["2026-01-01T00:00Z"])
        else:
            values[name] = pd.Series([1.0], dtype=dtype)
    if extra:
        values[extra] = [1]
    frame = pd.DataFrame(values)
    if version is not None:
        frame.attrs["weather_features_version"] = version
    return frame


def spec_for(path: Path, frame: pd.DataFrame) -> dict:
    spec = dict(PRODUCTION)
    spec.update(bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(frame),
                airports=sorted(frame["airport"].unique().tolist()),
                schema=[{"name": c, "dtype": str(frame[c].dtype)} for c in frame.columns],
                weather_features_version=frame.attrs.get("weather_features_version"),
                version_representation="synthetic fixture pandas attrs")
    return spec


class Era5DataCustodyTest(unittest.TestCase):
    def test_structure_version_schema_and_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.parquet"
            frame = payload()
            frame.to_parquet(path, index=False)
            spec = spec_for(path, frame)
            self.assertIsNone(verify_file(path, spec)["weather_features_version"])
            with self.assertRaisesRegex(ValueError, "version"):
                verify_file(path, dict(spec, weather_features_version=1))
            with self.assertRaisesRegex(ValueError, "schema"):
                verify_file(path, dict(spec, schema=list(reversed(spec["schema"]))))
            conflicting = payload(version=2)
            conflicting.to_parquet(path, index=False)
            conflicting_spec = spec_for(path, conflicting)
            with self.assertRaisesRegex(ValueError, "version"):
                verify_file(path, dict(conflicting_spec, weather_features_version=None))
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "custody mismatch"):
                verify_file(path, conflicting_spec)

    def test_generalized_competition_column_rejection(self) -> None:
        for column in ("movement_id", "private_TARGET_value", "flight_label_code",
                       "predicted_taxitime_seconds", "row_mvt_id_raw"):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "table.parquet"
                frame = payload(column)
                frame.to_parquet(path, index=False)
                with self.assertRaisesRegex(ValueError, "competition-row"):
                    verify_file(path, spec_for(path, frame))

    def test_bundle_exact_files_notice_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            frame = payload()
            data = bundle / PRODUCTION["file"]
            frame.to_parquet(data, index=False)
            spec = spec_for(data, frame)
            (bundle / MANIFEST_PATH.name).write_text(json.dumps(spec))
            (bundle / "NOTICE.txt").write_text(NOTICE)
            self.assertEqual(verify_bundle(bundle, spec)["rows"], 1)
            (bundle / "extra.txt").write_text("unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected bundle"):
                verify_bundle(bundle, spec)

    def test_production_cli_missing_bundle_and_pack_hash_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            verify = subprocess.run(
                [sys.executable, str(ROOT / "release/era5_data.py"), "verify", str(missing)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(verify.returncode, 0)
            self.assertIn("bundle absent", verify.stderr)
            source = Path(directory) / "synthetic.parquet"
            payload().to_parquet(source, index=False)
            pack = subprocess.run(
                [sys.executable, str(ROOT / "release/era5_data.py"), "pack", str(source),
                 "--out", str(Path(directory) / "bundle")], cwd=ROOT,
                capture_output=True, text=True)
            self.assertNotEqual(pack.returncode, 0)
            self.assertIn("custody mismatch", pack.stderr)

    def test_weather_loader_is_lazy(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import prc.weather_features as w; print(w.VERSION)"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2")

    def test_loader_discovers_override_before_reading(self) -> None:
        import os
        from unittest import mock
        import prc.weather_features as weather

        with tempfile.TemporaryDirectory() as directory:
            chosen = Path(directory) / "era5.parquet"
            chosen.touch()
            with mock.patch.dict(os.environ, {"PRC_ERA5_HOURLY": str(chosen)}):
                self.assertEqual(weather._default_hourly_path(), chosen)


if __name__ == "__main__":
    unittest.main()
