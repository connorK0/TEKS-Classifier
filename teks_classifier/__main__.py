"""Command-line entry point: python -m teks_classifier [--dry-run]"""

import argparse
import sys

from canvasapi import Canvas

from .classifier import TeksClassifier
from .config import Config, ConfigError
from .pipeline import HEADER, build_rows
from .sheets import open_worksheet, uploaded_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="teks_classifier",
        description="Suggest US History TEKS for new Canvas assignments and add them to a Google Sheet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print predictions for every assignment without reading or writing the sheet",
    )
    args = parser.parse_args()

    try:
        config = Config.from_env(require_sheets=not args.dry_run)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 1

    worksheet = None
    skip_ids = set()
    if not args.dry_run:
        worksheet = open_worksheet(config.service_account_file, config.spreadsheet_id)
        skip_ids = uploaded_ids(worksheet)

    course = Canvas(config.canvas_url, config.canvas_token).get_course(config.course_id)
    classifier = TeksClassifier(config.model_id)
    rows = build_rows(course.get_assignments(), classifier, skip_ids)

    for row in rows:
        print(" | ".join(row))
    print(f"{len(rows)} new assignment(s)")

    if worksheet is not None and rows:
        if not skip_ids:
            rows.insert(0, HEADER)
        # One API call for the whole batch. Nothing is recorded locally, so if
        # this fails the same rows are simply retried on the next run.
        worksheet.append_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
