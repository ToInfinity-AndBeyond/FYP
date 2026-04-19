#!/usr/bin/env python3
"""Download only the WFDB files listed in a MIMIC-III-Ext-PPG manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path


DEFAULT_MANIFEST = Path("artifacts/mimic_ext_subject_selection/selected_segments.csv")
DEFAULT_DATASET_ROOT = Path("/vol/bitbucket/mc1920/FYP/physionet.org/files/mimic-iii-ext-ppg/1.1.0")
DEFAULT_BASE_URL = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected WFDB files from MIMIC-III-Ext-PPG.")
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on manifest rows.")
    parser.add_argument(
        "--download-tool",
        choices=("curl", "wget"),
        default="curl",
        help="Transport backend. Use wget for interactive --ask-password auth.",
    )
    parser.add_argument("--user", type=str, default=None, help="Optional username for wget/curl auth.")
    parser.add_argument(
        "--ask-password",
        action="store_true",
        help="When using wget, prompt interactively for the password instead of storing it anywhere.",
    )
    parser.add_argument(
        "--netrc-file",
        type=Path,
        default=None,
        help="Optional netrc file containing PhysioNet credentials. Best for batch downloads with Basic auth.",
    )
    parser.add_argument("--cookie-file", type=Path, default=None, help="Optional cookie jar for PhysioNet auth.")
    parser.add_argument("--header", action="append", default=[], help="Optional extra HTTP header.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--abort-on-auth-failure", action="store_true", default=True)
    parser.add_argument("--output-log", type=Path, default=Path("artifacts/mimic_ext_subject_selection/download_log.json"))
    return parser.parse_args()


def build_curl_command(
    url: str,
    output_path: Path,
    cookie_file: Path | None,
    headers: list[str],
    user: str | None,
    netrc_file: Path | None,
) -> list[str]:
    command = ["curl", "-fL", "--retry", "3", "--retry-delay", "2", "-o", str(output_path)]
    if netrc_file is not None:
        command.extend(["--netrc-file", str(netrc_file)])
    elif user is not None:
        command.extend(["-u", f"{user}:"])
    if cookie_file is not None:
        command.extend(["-b", str(cookie_file), "-c", str(cookie_file)])
    for header in headers:
        command.extend(["-H", header])
    command.append(url)
    return command


def build_wget_command(
    url: str,
    output_path: Path,
    user: str | None,
    ask_password: bool,
    netrc_file: Path | None,
    cookie_file: Path | None,
    headers: list[str],
) -> list[str]:
    command = ["wget", "-q", "-O", str(output_path), "-c", "-N"]
    if user:
        command.extend(["--user", user])
    if ask_password:
        command.append("--ask-password")
    if cookie_file is not None:
        command.extend(["--load-cookies", str(cookie_file), "--save-cookies", str(cookie_file), "--keep-session-cookies"])
    for header in headers:
        command.extend(["--header", header])
    command.append(url)
    return command


def resolve_wfdb_record_path(row: dict[str, str]) -> str:
    explicit_path = str(row.get("wfdb_record_path", "")).strip()
    if explicit_path:
        return explicit_path
    folder = Path(str(row["folder_path"]))
    signal_name = str(row["signal_file_name"])
    if folder.name == signal_name:
        return folder.as_posix()
    return (folder / signal_name).as_posix()


def main() -> None:
    args = parse_args()
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.manifest_csv.open("r", encoding="utf-8", newline="")))
    if args.limit is not None:
        rows = rows[: args.limit]

    log_entries = []
    status_counter = Counter()

    for row in rows:
        rel_base = resolve_wfdb_record_path(row)
        for ext in (".hea", ".dat"):
            rel_path = f"{rel_base}{ext}"
            target_path = args.dataset_root / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if target_path.exists() and target_path.stat().st_size > 0:
                status_counter["already_present"] += 1
                log_entries.append({"path": rel_path, "status": "already_present"})
                continue

            url = f"{args.base_url}/{rel_path}"
            if args.dry_run:
                status_counter["dry_run"] += 1
                log_entries.append({"path": rel_path, "status": "dry_run", "url": url})
                continue

            if args.download_tool == "wget":
                command = build_wget_command(
                    url,
                    target_path,
                    args.user,
                    args.ask_password,
                    args.netrc_file,
                    args.cookie_file,
                    args.header,
                )
            else:
                command = build_curl_command(
                    url,
                    target_path,
                    args.cookie_file,
                    args.header,
                    args.user,
                    args.netrc_file,
                )
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and target_path.exists() and target_path.stat().st_size > 0:
                status_counter["downloaded"] += 1
                log_entries.append({"path": rel_path, "status": "downloaded", "url": url})
                continue

            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            status = "failed"
            if "403" in stderr or "403" in stdout:
                status = "auth_failed"
            elif "404" in stderr or "404" in stdout:
                status = "not_found"
            status_counter[status] += 1
            log_entries.append(
                {
                    "path": rel_path,
                    "status": status,
                    "url": url,
                    "returncode": result.returncode,
                    "stderr": stderr[-500:],
                    "stdout": stdout[-500:],
                }
            )

            if target_path.exists() and target_path.stat().st_size == 0:
                target_path.unlink()

            if status == "auth_failed" and args.abort_on_auth_failure:
                summary = {
                    "manifest_rows": len(rows),
                    "file_status_counts": dict(status_counter),
                    "message": "Credentialed access failed. Supply PhysioNet auth cookies, headers, or rerun with wget --user/--ask-password.",
                }
                args.output_log.write_text(
                    json.dumps({"summary": summary, "entries": log_entries}, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(summary, indent=2))
                return

    summary = {
        "manifest_rows": len(rows),
        "file_status_counts": dict(status_counter),
    }
    args.output_log.write_text(
        json.dumps({"summary": summary, "entries": log_entries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
