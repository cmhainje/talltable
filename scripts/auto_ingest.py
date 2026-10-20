"""
auto_ingest.py

detects if new, uningested SPHEREx data exists in the data dir
submits an ingestion job if so
also maintains a manifest of submitted jobs, statuses, etc in slurm/manifest.json

each run looks at the data directory of one release (--release, default qr3): TALLTABLE_DATA_DIR_QR3 for qr3,
TALLTABLE_DATA_DIR for qr2. the job it submits gets that directory as its TALLTABLE_DATA_DIR, so everything the
job runs (ingest, compact, build_wcs, ...) works on that release without further changes.

a week that has already been ingested is re-checked on every run: if more files have shown up since (for
example an `l2b_retry-*` directory that arrived after the main directory), the week is reopened and
ingested again. ingest.py skips images it has already ingested, so only the new ones are added.

usage:
    uv run python scripts/auto_ingest.py              # normal run
    uv run python scripts/auto_ingest.py --dry-run    # don't submit or write manifest
    uv run python scripts/auto_ingest.py --bootstrap  # one-time: seed manifest from DB
    uv run python scripts/auto_ingest.py --force 2026W36  # submit a week now, skipping the stability gate

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

# the environment variable that holds the data directory of each release
RELEASE_DATA_DIR_VARS = {"qr2": "TALLTABLE_DATA_DIR", "qr3": "TALLTABLE_DATA_DIR_QR3"}
DEFAULT_RELEASE = "qr3"
SLURM_DIR = REPO_ROOT / "slurm"
SBATCH_TEMPLATE = SLURM_DIR / "ingest_week.sbatch"
SLURM_FILES_DIR = SLURM_DIR / "files"  # ingest_week.sbatch looks for {year}w{week}-files.txt here
MANIFEST_PATH = SLURM_DIR / "manifest.json"
LOG_PATH = SLURM_DIR / "auto_ingest.log"

# e.g., 2026W15_2A
WEEK_DIR_RE = re.compile(r"^(\d{4})W(\d{2})_")

# lines that trip these are recorded as warnings, except the expected DeprecationWarning
WARNING_RE = re.compile(r"traceback|error|warning", re.IGNORECASE)
IGNORED_WARNING_RE = re.compile(r"deprecationwarning", re.IGNORECASE)

# a week is safe to ingest once its file count has been unchanged for at least this long
STABILITY_MIN_GAP = timedelta(hours=20)

# ...or once no FITS file has arrived in this long (see is_stable)
NO_NEW_FILES_THRESHOLD = timedelta(days=7)

MAX_WARNINGS_STORED = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def week_key(year: int, week: int) -> str:
    return f"{year}W{week:02d}"


def entry_release(entry: dict) -> str:
    """the release of a manifest entry. entries from before releases were tracked are all QR2."""
    return entry.get("release", "qr2")


def release_level2_dir(release: str) -> Path:
    """the `level2` directory of a release's data directory"""
    return require_env(RELEASE_DATA_DIR_VARS[release]) / "level2"


def count_fits(data_dir: Path, folders: list[str]) -> int:
    """number of FITS files in the given week folders: the same files as ingest_week.sbatch's `find`
    (two directory levels under each week folder), counted without stat'ing each one"""
    return sum(1 for d in folders for _ in (data_dir / d).glob("*/*/*.fits"))


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


def bootstrap_manifest(data_dir: Path, release: str) -> dict:
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
    week_release = {}
    for fp in filepaths:
        m = re.search(r"(\d{4})W(\d{2})_", str(fp))
        if not m:
            continue
        year, week = int(m.group(1)), int(m.group(2))
        weeks_seen.setdefault((year, week), 0)
        weeks_seen[(year, week)] += 1
        # the release is in the filepath (.../spherex_data_qr3/level2/...); `release` is the fallback
        r = re.search(r"qr\d+", str(fp))
        week_release.setdefault((year, week), r.group(0) if r else release)

    for (year, week), count in sorted(weeks_seen.items()):
        folders = sorted(
            e.name for e in data_dir.iterdir()
            if e.is_dir() and (m := WEEK_DIR_RE.match(e.name)) and (int(m.group(1)), int(m.group(2))) == (year, week)
        ) if data_dir.exists() else []

        manifest[week_key(year, week)] = {
            "year": year,
            "week": week,
            "release": week_release[(year, week)],
            "folders": folders,
            "file_counts_by_check": [],
            "file_count_at_submit": count,  # the baseline that new files are detected against
            "latest_file_ctime": None,
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


def reopen_if_new_files(key: str, entry: dict, data_dir: Path, folders: list[str]) -> None:
    """a week that has been ingested: if there are more files than were ingested (for example an
    `l2b_retry-*` directory that arrived after the main one), reopen it ("detected") so that it is
    ingested again. ingest.py skips images that are already in the database, so only the new ones are added.

    the baseline is `file_count_at_submit`: the number of files in the list the ingest job was given."""
    baseline = entry.get("file_count_at_submit")
    if baseline is None:
        return  # nothing to compare against (weeks seeded by --bootstrap record one)

    count = count_fits(data_dir, folders)
    if count < baseline:
        logger.warning("%s: %d files now, but %d were ingested; files were removed?", key, count, baseline)
        return
    if count == baseline:
        if sorted(folders) != entry["folders"]:
            logger.info("%s: folder set changed (was %s, now %s) but no new files", key, entry["folders"], sorted(folders))
            entry["folders"] = sorted(folders)
        return

    logger.info(
        "%s: %d new file(s) since the previous ingest (%d -> %d; folders %s -> %s); reopening",
        key, count - baseline, baseline, count, entry["folders"], sorted(folders),
    )
    entry.setdefault("previous_runs", []).append({
        "job_id": entry["job_id"],
        "job_name": entry["job_name"],
        "log_path": entry["log_path"],
        "submitted_at": entry["submitted_at"],
        "completed_at": entry["completed_at"],
        "status": entry["status"],
        "file_count": baseline,
        "warnings": entry["warnings"],
    })
    entry.update(
        status="detected",
        folders=sorted(folders),
        file_counts_by_check=[],
        latest_file_ctime=None,
        job_id=None,
        job_name=None,
        log_path=None,
        submitted_at=None,
        completed_at=None,
        warnings=[],
        notes=f"reopened: {count - baseline} new file(s) after the previous ingest",
        reopened_at=now_iso(),
    )


def detect_new_weeks(manifest: dict, data_dir: Path, release: str) -> None:
    """glob level2, record file-count samples for unsubmitted weeks.
    weeks that were already ingested are checked for files that arrived later (see reopen_if_new_files)."""
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

        if entry is not None and entry_release(entry) != release:
            logger.warning(
                "%s is in the manifest as a %s week, but is also in the %s data directory: skipping it "
                "(a reprocessed release needs its own handling)",
                key, entry_release(entry), release,
            )
            continue

        if entry is None:
            entry = {
                "year": year,
                "week": week,
                "release": release,
                "folders": sorted(folders),
                "file_counts_by_check": [],
                "file_count_at_submit": None,
                "latest_file_ctime": None,
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

        if entry["status"] in ("done", "done_with_warnings"):
            reopen_if_new_files(key, entry, data_dir, folders)

        if entry["status"] != "detected":
            if sorted(folders) != entry["folders"]:
                # (an ingest that is in flight, or a failed one: whatever arrived is picked up as new
                # files once the week has completed)
                logger.warning(
                    "%s: folder set changed since the last check! was %s, now %s (status %s)",
                    key, entry["folders"], sorted(folders), entry["status"],
                )
            continue

        if len(folders) == 1 and not entry["notes"].startswith("reopened"):
            entry["notes"] = f"only 1 folder found: {sorted(folders)}"
            logger.warning("%s: %s", key, entry["notes"])

        entry["folders"] = sorted(folders)

        # matches ingest_week.sbatch's `find -mindepth 4 -maxdepth 4` relative to
        # data_dir: two subdirectory levels under each week folder, then the file
        count = 0
        latest_ctime = 0.0
        filepaths = []
        for d in folders:
            for f in (data_dir / d).glob("*/*/*.fits"):
                count += 1
                latest_ctime = max(latest_ctime, f.stat().st_ctime)
                filepaths.append(str(f))

        entry["file_counts_by_check"].append({"ts": now_iso(), "count": count})
        entry.pop("latest_file_mtime", None)  # replaced by latest_file_ctime, see is_stable
        if latest_ctime:
            entry["latest_file_ctime"] = datetime.fromtimestamp(latest_ctime, tz=timezone.utc).isoformat()

        # we're already paying the cost of walking every file for the count/ctime
        # above, so write out ingest_week.sbatch's file list here for free -- it
        # checks for this file and skips its own (much slower, serial) `find` if
        # it's already present
        if filepaths:
            SLURM_FILES_DIR.mkdir(parents=True, exist_ok=True)
            filelist_path = SLURM_FILES_DIR / f"{year}w{week:02d}-files.txt"
            filelist_path.write_text("\n".join(sorted(filepaths)) + "\n")


def is_stable(entry: dict) -> bool:
    """ready to ingest once either:
        (a) the file count has been the same at every check for at least STABILITY_MIN_GAP
            (from the first check that saw the current count to the latest one)
        (b) no file has arrived in over NO_NEW_FILES_THRESHOLD

    (b) uses ctime, not mtime: the S3 download (`aws s3 sync`/`cp`) sets each file's mtime to the
    object's LastModified, which can be days before the file reached this cluster. ctime can't be set
    that way, so it is when the file was written here (or later, if its metadata changed since)."""
    latest_ctime = entry.get("latest_file_ctime")
    if latest_ctime is not None:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(latest_ctime)
        if age >= NO_NEW_FILES_THRESHOLD:
            return True

    samples = entry["file_counts_by_check"]
    if not samples or samples[-1]["count"] == 0:
        return False
    count = samples[-1]["count"]
    first = len(samples) - 1
    while first > 0 and samples[first - 1]["count"] == count:
        first -= 1
    ts_first = datetime.fromisoformat(samples[first]["ts"])
    ts_last = datetime.fromisoformat(samples[-1]["ts"])
    return (ts_last - ts_first) >= STABILITY_MIN_GAP


def _submit(entry: dict, dry_run: bool, data_dir: Path) -> None:
    """sbatch the ingest job for a single manifest entry and update it in place.

    `data_dir` is the release's `level2` directory. the job gets its parent as TALLTABLE_DATA_DIR (which
    takes precedence over .env), so ingest, compact, build_wcs, ... all work on that release."""
    year, week = entry["year"], entry["week"]
    week_str = f"{week:02d}"
    job_name = f"ingest{year}W{week_str}"
    cmd = [
        "sbatch",
        "--job-name", job_name,
        "--export", f"ALL,YEAR={year},WEEK={week_str},TALLTABLE_DATA_DIR={data_dir.parent}",
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
    # the file list the job uses is the one written by the latest detection pass: this many files.
    # (a later pass that finds more files than this reopens the week, see reopen_if_new_files)
    samples = entry["file_counts_by_check"]
    entry["file_count_at_submit"] = samples[-1]["count"] if samples else None
    entry["submitted_at"] = now_iso()
    entry["job_id"] = job_id
    entry["job_name"] = job_name
    entry["log_path"] = str(SLURM_DIR / f"slurm-{job_name}-{job_id}.out")
    logger.info("submitted %s as job %s", week_key(year, week), job_id)


def submission_blocked(manifest: dict) -> bool:
    """whether a new submission must wait: a week has failed, or an ingest job is already in flight.

    note: NEVER run more than one at a time: post_compact.py will NOT handle that gracefully"""
    failed = [e for e in manifest.values() if e["status"] == "failed"]
    if failed:
        logger.error(
            "week %s FAILED (job %s); auto-submission halted until fixed. "
            "retry with --retry-failed",
            week_key(failed[0]["year"], failed[0]["week"]), failed[0]["job_id"],
        )
        return True

    in_flight = [e for e in manifest.values() if e["status"] == "submitted"]
    if in_flight:
        logger.info(
            "week %s is already submitted (job %s); skipping new submissions this run",
            week_key(in_flight[0]["year"], in_flight[0]["week"]), in_flight[0]["job_id"],
        )
        return True

    return False


def submit_ready_week(manifest: dict, dry_run: bool, data_dir: Path, release: str) -> None:
    """submit a week's ingest job (of the given release)."""
    if submission_blocked(manifest):
        return

    candidates = sorted(
        (
            e for e in manifest.values()
            if e["status"] == "detected" and entry_release(e) == release and is_stable(e)
        ),
        key=lambda e: (e["year"], e["week"]),
    )
    if not candidates:
        return

    _submit(candidates[0], dry_run, data_dir)


def force_submit_week(manifest: dict, key: str, dry_run: bool, data_dir: Path, release: str) -> None:
    """submit the given week now, whether or not it is stable yet. the one-job-at-a-time and
    halt-on-failure rules still apply."""
    entry = manifest.get(key)
    if entry is None:
        logger.error("--force %s: no such week in the manifest (not in %s?)", key, data_dir)
        return
    if entry_release(entry) != release:
        logger.error("--force %s: it is a %s week; rerun with --release %s", key, entry_release(entry), entry_release(entry))
        return
    if entry["status"] != "detected":
        logger.error("--force %s: status is %s, not detected", key, entry["status"])
        return
    if submission_blocked(manifest):
        return

    if not is_stable(entry):
        logger.info("%s: forcing submission past the stability gate (count %s)", key, entry["file_counts_by_check"][-1]["count"])
        if not dry_run:
            entry["forced_at"] = now_iso()
    _submit(entry, dry_run, data_dir)


def retry_failed_week(manifest: dict, dry_run: bool, data_dir: Path | None) -> None:
    """re-submit the failed week. (`data_dir`: the level2 directory to ingest from, by default the
    one of the failed week's release)"""
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

    _submit(entry, dry_run, data_dir or release_level2_dir(entry_release(entry)))


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
    parser.add_argument(
        "--release", choices=sorted(RELEASE_DATA_DIR_VARS), default=DEFAULT_RELEASE,
        help="the release to ingest. its data directory is TALLTABLE_DATA_DIR_QR3 (qr3) or TALLTABLE_DATA_DIR (qr2)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="the `level2` directory to look in (default: the release's data directory + /level2)",
    )
    parser.add_argument(
        "--force", metavar="YYYYWww",
        help="submit this week now even if it isn't stable yet (still never while another job is in flight)",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    data_dir = args.data_dir

    if args.retry_failed:
        manifest = load_manifest(args.manifest)
        retry_failed_week(manifest, args.dry_run, data_dir)
        if not args.dry_run:
            save_manifest(args.manifest, manifest)
        return

    if data_dir is None:
        data_dir = release_level2_dir(args.release)

    if args.bootstrap:
        if args.manifest.exists():
            logger.error("%s already exists; refusing to overwrite. Remove it first if you really want to re-bootstrap.", args.manifest)
            return
        manifest = bootstrap_manifest(data_dir, args.release)
        save_manifest(args.manifest, manifest)
        logger.info("wrote %d entries to %s", len(manifest), args.manifest)
        return

    manifest = load_manifest(args.manifest)
    # job status first: a week whose job has just completed is then checked for late files in the same pass
    check_job_status(manifest)
    detect_new_weeks(manifest, data_dir, args.release)
    if args.force:
        force_submit_week(manifest, args.force.upper(), args.dry_run, data_dir, args.release)
    else:
        submit_ready_week(manifest, args.dry_run, data_dir, args.release)

    if not args.dry_run:
        save_manifest(args.manifest, manifest)
    else:
        logger.info("[dry-run] not writing manifest")


if __name__ == "__main__":
    main()
