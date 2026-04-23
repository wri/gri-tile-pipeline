#!/usr/bin/env python3
"""Merge new IAM statements into an existing S3 bucket policy.

Used by docs/manual_wri_policy_update.md to add the Lambda-role cross-account
grant to tof-output's bucket policy in the wri account without clobbering
pre-existing statements.

Inputs:
    --current   Path to the current full policy JSON (from `aws s3api get-bucket-policy`).
    --append    Path to a JSON array of statement objects (from `terraform output
                cross_account_policy_statements_json` piped through `jq fromjson`,
                or written to a file).
    --out       Path to write the merged full policy JSON.

Behavior:
    - Reads --current, parses as an S3 bucket policy document.
    - Reads --append, parses as a list of statement objects.
    - For each appended statement: if an existing statement has the same Sid,
      replace it; otherwise append. Makes the merge idempotent.
    - Validates: merged doc has Version + Statement[], total size < 20 KB
      (S3 bucket policy hard limit).
    - Writes pretty-printed JSON to --out.
    - Prints a human summary: which SIDs were added, replaced, untouched.

Exit codes:
    0 on success. Non-zero (with a clear message on stderr) on validation failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

S3_BUCKET_POLICY_MAX_BYTES = 20 * 1024


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        sys.exit(f"error: {path} does not exist")
    except json.JSONDecodeError as e:
        sys.exit(f"error: {path} is not valid JSON: {e}")


def _validate_current(doc: object, path: Path) -> dict:
    if not isinstance(doc, dict):
        sys.exit(f"error: {path} must be a JSON object, got {type(doc).__name__}")
    if "Statement" not in doc or not isinstance(doc["Statement"], list):
        sys.exit(f"error: {path} must have a 'Statement' array")
    doc.setdefault("Version", "2012-10-17")
    return doc


def _validate_append(statements: object, path: Path) -> list[dict]:
    if not isinstance(statements, list):
        sys.exit(f"error: {path} must be a JSON array of statements, got {type(statements).__name__}")
    required = {"Effect", "Principal", "Action", "Resource"}
    for i, s in enumerate(statements):
        if not isinstance(s, dict):
            sys.exit(f"error: {path}[{i}] must be an object")
        missing = required - s.keys()
        if missing:
            sys.exit(f"error: {path}[{i}] missing required fields: {sorted(missing)}")
        if "Sid" not in s:
            sys.exit(f"error: {path}[{i}] must have a 'Sid' (required for idempotent merges)")
    return statements


def merge(current: dict, appended: list[dict]) -> tuple[dict, dict[str, list[str]]]:
    """Return (merged_policy, summary). summary keys: added, replaced, untouched."""
    existing = list(current["Statement"])
    sid_to_idx = {s.get("Sid"): i for i, s in enumerate(existing) if s.get("Sid")}

    added: list[str] = []
    replaced: list[str] = []
    for s in appended:
        sid = s["Sid"]
        if sid in sid_to_idx:
            existing[sid_to_idx[sid]] = s
            replaced.append(sid)
        else:
            existing.append(s)
            added.append(sid)

    merged = {"Version": current.get("Version", "2012-10-17"), "Statement": existing}
    appended_sids = {s["Sid"] for s in appended}
    untouched = [s.get("Sid", "<no-sid>") for s in current["Statement"] if s.get("Sid") not in appended_sids]
    summary = {"added": added, "replaced": replaced, "untouched": untouched}
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", required=True, type=Path, help="Existing policy JSON")
    parser.add_argument("--append", required=True, type=Path, help="Statements JSON array to append")
    parser.add_argument("--out", required=True, type=Path, help="Write merged policy here")
    args = parser.parse_args()

    current = _validate_current(_load_json(args.current), args.current)
    appended = _validate_append(_load_json(args.append), args.append)

    merged, summary = merge(current, appended)
    out_text = json.dumps(merged, indent=2, sort_keys=False) + "\n"

    size = len(out_text.encode("utf-8"))
    if size > S3_BUCKET_POLICY_MAX_BYTES:
        sys.exit(
            f"error: merged policy is {size} bytes, exceeds S3's {S3_BUCKET_POLICY_MAX_BYTES}-byte limit"
        )

    args.out.write_text(out_text)

    print(f"Merged policy written to {args.out} ({size} bytes)")
    print(f"  Added SIDs:     {summary['added'] or '(none)'}")
    print(f"  Replaced SIDs:  {summary['replaced'] or '(none)'}")
    print(f"  Untouched SIDs: {summary['untouched'] or '(none)'}")
    print()
    print("Next: diff the result, then apply with aws s3api put-bucket-policy.")


if __name__ == "__main__":
    main()
