"""Reads from and appends to the Google Sheet that teachers review."""

import gspread

ID_COLUMN = 2  # 1-based column index of "Canvas ID", matching pipeline.HEADER


def open_worksheet(service_account_file: str, spreadsheet_id: str) -> gspread.Worksheet:
    """Open the first tab of the sheet.

    Uses a service account so the job can run unattended. The sheet must be
    shared with the service account's email address.
    """
    client = gspread.service_account(filename=service_account_file)
    return client.open_by_key(spreadsheet_id).sheet1


def uploaded_ids(worksheet: gspread.Worksheet) -> set[str]:
    """Canvas IDs already in the sheet. The sheet is the only record of what's been done."""
    return set(worksheet.col_values(ID_COLUMN))
