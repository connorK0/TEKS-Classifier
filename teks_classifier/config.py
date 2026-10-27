"""Settings come from environment variables (or a local .env file), never from source code."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Fine-tuned DistilBERT published on the Hugging Face Hub.
DEFAULT_MODEL = "connork0/10Batch25Epoch2Grad"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    canvas_url: str
    canvas_token: str
    course_id: int
    spreadsheet_id: str | None
    service_account_file: str | None
    model_id: str

    @classmethod
    def from_env(cls, require_sheets: bool = True) -> "Config":
        load_dotenv()
        required = ["CANVAS_URL", "CANVAS_TOKEN", "CANVAS_COURSE_ID"]
        if require_sheets:
            required += ["GOOGLE_SHEET_ID", "GOOGLE_SERVICE_ACCOUNT_FILE"]
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ConfigError(f"Missing environment variables: {', '.join(missing)} (see .env.example)")

        return cls(
            canvas_url=os.environ["CANVAS_URL"],
            canvas_token=os.environ["CANVAS_TOKEN"],
            course_id=int(os.environ["CANVAS_COURSE_ID"]),
            spreadsheet_id=os.environ.get("GOOGLE_SHEET_ID"),
            service_account_file=os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE"),
            model_id=os.environ.get("TEKS_MODEL", DEFAULT_MODEL),
        )
