"""
auto_ingest.py

detects if new, uningested SPHEREx data exists in the data dir
submits an ingestion job if so
also maintains a manifest of submitted jobs, statuses, etc in slurm/manifest.json

usage:
    uv run python scripts/auto_ingest.py              # normal run
    uv run python scripts/auto_ingest.py --dry-run    # don't submit or write manifest
    uv run python scripts/auto_ingest.py --bootstrap  # one-time: seed manifest from DB

suggested crontab line:

    0 9 * * * cd /mnt/home/chainje/spxperiments && ./.venv/bin/python scripts/auto_ingest.py >> slurm/auto_ingest_cron.log 2>&1
"""

import json
import logging
import re
import subprocess

from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from pathlib import Path
from talltable.paths import require_env

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = require_env("TALLTABLE_DATA_DIR") / "level2"
SLURM_DIR = REPO_ROOT / "slurm"
SBATCH_TEMPLATE = SLURM_DIR / "ingest_week.sbatch"
MANIFEST_PATH = SLURM_DIR / "manifest.json"
LOG_PATH = SLURM_DIR / "auto_ingest.log"

# e.g., 2026W15_2A
WEEK_DIR_RE = re.compile(r"^(\d{4})W(\d{2})_")

# lines that trip these are recorded as warnings, except the expected DeprecationWarning
WARNING_RE = re.compile(r"traceback|error|warning", re.IGNORECASE)
IGNORED_WARNING_RE = re.compile(r"deprecationwarning", re.IGNORECASE)

# a week is safe to ingest once its file count is unchanged
# for two checks at least this far apart
STABILITY_MIN_GAP = timedelta(hours=20)

MAX_WARNINGS_STORED = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def week_key(year: int, week: int) -> str:
    return f"{year}W{week:02d}"


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp.replace(path)


def bootstrap_manifest(data_dir: Path) -> dict:
    """seed manifest from images table"""
    import duckdb
    from talltable.paths import IMAGE_DB_PATH

    manifest = {}
    if not IMAGE_DB_PATH.exists():
        logger.warning("no image table found at %s; starting empty manifest", IMAGE_DB_PATH)
        return manifest

    filepaths = duckdb.sql(
        f"SELECT DISTINCT filepath FROM read_parquet('{IMAGE_DB_PATH}')"
    ).fetchnumpy()["filepath"]

    weeks_seen = {}
    for fp in filepaths:
        m = re.search(r"(\d{4})W(\d{2})_", str(fp))
        if not m:
            continue
        year, week = int(m.group(1)), int(m.group(2))
        weeks_seen.setdefault((year, week), 0)
        weeks_seen[(year, week)] += 1

    for (year, week), count in sorted(weeks_seen.items()):
        folders = sorted(
            e.name for e in data_dir.iterdir()
            if e.is_dir() and (m := WEEK_DIR_RE.match(e.name)) and (int(m.group(1)), int(m.group(2))) == (year, week)
        ) if data_dir.exists() else []

        manifest[week_key(year, week)] = {
            "year": year,
            "week": week,
            "folders": folders,
            "file_counts_by_check": [],
            "status": "done",
            "detected_at": None,
            "submitted_at": None,
            "completed_at": None,
            "job_id": None,
            "job_name": None,
            "log_path": None,
            "warnings": [],
            "notes": "seeded from images table by --bootstrap",
        }
        logger.info("bootstrapped %s (%d images already in DB)", week_key(year, week), count)

    return manifest


def detect_new_weeks(manifest: dict, data_dir: Path) -> None:
    """glob level2, record file-count samples for unsubmitted weeks"""
    if not data_dir.exists():
        logger.error("data dir %s does not exist", data_dir)
        return

    weeks_found: dict[tuple[int, int], list[str]] = {}
    for entry in data_dir.iterdir():
        if not entry.is_dir():
            continue
        m = WEEK_DIR_RE.match(entry.name)
        if not m:
            continue
        year, week = int(m.group(1)), int(m.group(2))
        weeks_found.setdefault((year, week), []).append(entry.name)

    for (year, week), folders in sorted(weeks_found.items()):
        key = week_key(year, week)
        entry = manifest.get(key)

        if entry is None:
            entry = {
                "year": year,
                "week": week,
                "folders": sorted(folders),
                "file_counts_by_check": [],
                "status": "detected",
                "detected_at": now_iso(),
                "submitted_at": None,
                "completed_at": None,
                "job_id": None,
                "job_name": None,
                "log_path": None,
                "warnings": [],
                "notes": "",
            }
            manifest[key] = entry
            logger.info("detected new week %s (folders: %s)", key, sorted(folders))

        if entry["status"] != "detected":
            if sorted(folders) != entry["folders"]:
                logger.error(
                    "%s: folder set changed after submission! was %s, now %s",
                    key, entry["folders"], sorted(folders),
                )
            continue

        if len(folders) == 1:
            entry["notes"] = f"only 1 folder found: {sorted(folders)}"
            logger.warning("%s: %s", key, entry["notes"])

        entry["folders"] = sorted(folders)

        count = sum(1 for d in folders for _ in (data_dir / d).rglob("*.fits"))
        entry["file_counts_by_check"].append({"ts": now_iso(), "count": count})


def is_stable(entry: dict) -> bool:
    samples = entry["file_counts_by_check"]
    if len(samples) < 2:
        return False
    a, b = samples[-2], samples[-1]
    if a["count"] == 0 or a["count"] != b["count"]:
        return False
    ts_a = datetime.fromisoformat(a["ts"])
    ts_b = datetime.fromisoformat(b["ts"])
    return (ts_b - ts_a) >= STABILITY_MIN_GAP


def _submit(entry: dict, dry_run: bool) -> None:
    """sbatch the ingest job for a single manifest entry and update it in place."""
    year, week = entry["year"], entry["week"]
    week_str = f"{week:02d}"
    job_name = f"ingest{year}W{week_str}"
    cmd = [
        "sbatch",
        "--job-name", job_name,
        "--export", f"ALL,YEAR={year},WEEK={week_str}",
        str(SBATCH_TEMPLATE),
    ]

    if dry_run:
        logger.info("[dry-run] would submit: %s", " ".join(cmd))
        return

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        logger.error("could not parse job id from sbatch output: %s", result.stdout)
        return

    job_id = m.group(1)
    entry["status"] = "submitted"
    entry["submitted_at"] = now_iso()
    entry["job_id"] = job_id
    entry["job_name"] = job_name
    entry["log_path"] = str(SLURM_DIR / f"slurm-{job_name}-{job_id}.out")
    logger.info("submitted %s as job %s", week_key(year, week), job_id)


def submit_ready_week(manifest: dict, dry_run: bool) -> None:
    """submit a week's ingest job.

    note: NEVER run more than one at a time: post_compact.py will NOT handle that gracefully"""
    failed = [e for e in manifest.values() if e["status"] == "failed"]
    if failed:
        logger.error(
            "week %s FAILED (job %s); auto-submission halted until fixed. "
            "retry with --retry-failed",
            week_key(failed[0]["year"], failed[0]["week"]), failed[0]["job_id"],
        )
        return

    in_flight = [e for e in manifest.values() if e["status"] == "submitted"]
    if in_flight:
        logger.info(
            "week %s is already submitted (job %s); skipping new submissions this run",
            week_key(in_flight[0]["year"], in_flight[0]["week"]), in_flight[0]["job_id"],
        )
        return

    candidates = sorted(
        (e for e in manifest.values() if e["status"] == "detected" and is_stable(e)),
        key=lambda e: (e["year"], e["week"]),
    )
    if not candidates:
        return

    _submit(candidates[0], dry_run)


def retry_failed_week(manifest: dict, dry_run: bool) -> None:
    """re-submit the failed week"""
    failed = [e for e in manifest.values() if e["status"] == "failed"]
    if not failed:
        logger.info("no failed weeks to retry")
        return
    if len(failed) > 1:
        logger.error(
            "multiple failed weeks found (%s); refusing to guess which to retry -- "
            "this shouldn't happen since submission halts after the first failure. "
            "fix slurm/manifest.json by hand",
            [week_key(e["year"], e["week"]) for e in failed],
        )
        return

    entry = failed[0]
    key = week_key(entry["year"], entry["week"])
    logger.info("retrying %s (previous job %s failed: %s)", key, entry["job_id"], entry["notes"])

    history = entry.setdefault("previous_attempts", [])
    history.append({
        "job_id": entry["job_id"],
        "job_name": entry["job_name"],
        "log_path": entry["log_path"],
        "submitted_at": entry["submitted_at"],
        "completed_at": entry["completed_at"],
        "notes": entry["notes"],
        "warnings": entry["warnings"],
    })

    entry["status"] = "detected"
    entry["job_id"] = None
    entry["job_name"] = None
    entry["log_path"] = None
    entry["submitted_at"] = None
    entry["completed_at"] = None
    entry["notes"] = ""
    entry["warnings"] = []

    _submit(entry, dry_run)


def scan_log_for_warnings(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    warnings = []
    with open(log_path, errors="ignore") as f:
        for line in f:
            if IGNORED_WARNING_RE.search(line):
                continue
            if WARNING_RE.search(line):
                warnings.append(line.strip()[:300])
                if len(warnings) >= MAX_WARNINGS_STORED:
                    break
    return warnings


def check_job_status(manifest: dict) -> None:
    for entry in manifest.values():
        if entry["status"] != "submitted":
            continue

        job_id = entry["job_id"]
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State,ExitCode", "-n", "-P", "-X"],
            capture_output=True, text=True,
        )
        line = result.stdout.strip()
        if not line:
            logger.info("job %s not yet in accounting; skipping", job_id)
            continue

        state, exit_code = line.split("|")
        if state in ("PENDING", "RUNNING", "REQUEUED", "COMPLETING", "SUSPENDED"):
            continue

        entry["completed_at"] = now_iso()

        if state != "COMPLETED" or exit_code != "0:0":
            entry["status"] = "failed"
            entry["notes"] = f"slurm state={state} exit_code={exit_code}"
            logger.warning("%s: job %s failed (%s)", week_key(entry["year"], entry["week"]), job_id, entry["notes"])
            continue

        warnings = scan_log_for_warnings(Path(entry["log_path"]))
        entry["warnings"] = warnings
        entry["status"] = "done_with_warnings" if warnings else "done"
        logger.info(
            "%s: job %s completed (%s)",
            week_key(entry["year"], entry["week"]), job_id, entry["status"],
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="detect and check status, but don't submit or save the manifest")
    parser.add_argument("--bootstrap", action="store_true", help="seed manifest.json from the images table and exit")
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="resubmit the failed week (only after you've fixed whatever caused it to fail) and exit",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    if args.bootstrap:
        if args.manifest.exists():
            logger.error("%s already exists; refusing to overwrite. Remove it first if you really want to re-bootstrap.", args.manifest)
            return
        manifest = bootstrap_manifest(args.data_dir)
        save_manifest(args.manifest, manifest)
        logger.info("wrote %d entries to %s", len(manifest), args.manifest)
        return

    if args.retry_failed:
        manifest = load_manifest(args.manifest)
        retry_failed_week(manifest, args.dry_run)
        if not args.dry_run:
            save_manifest(args.manifest, manifest)
        return

    manifest = load_manifest(args.manifest)
    detect_new_weeks(manifest, args.data_dir)
    check_job_status(manifest)
    submit_ready_week(manifest, args.dry_run)

    if not args.dry_run:
        save_manifest(args.manifest, manifest)
    else:
        logger.info("[dry-run] not writing manifest")


if __name__ == "__main__":
    main()
