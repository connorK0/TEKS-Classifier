"""Turns Canvas assignments into spreadsheet rows of TEKS predictions."""

from collections.abc import Container, Iterable
from typing import Protocol

from .lesson_parser import lesson_text

HEADER = ["Assignment", "Canvas ID", "Prediction 1", "Prediction 2", "Prediction 3"]


class Assignment(Protocol):
    id: int
    name: str
    description: str | None


class Classifier(Protocol):
    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]: ...


def format_prediction(label: str, probability: float) -> str:
    return f"{label} ({probability:.0%})"


def build_rows(
    assignments: Iterable[Assignment],
    classifier: Classifier,
    skip_ids: Container[str] = frozenset(),
) -> list[list[str]]:
    """Predict TEKS for every assignment not already in `skip_ids`.

    IDs are compared as strings: Canvas returns ints, but anything read back
    from a spreadsheet or JSON file is a string.
    """
    rows = []
    for assignment in assignments:
        assignment_id = str(assignment.id)
        if assignment_id in skip_ids:
            continue
        text = lesson_text(assignment.description)
        if not text:
            print(f"Skipping {assignment.name!r}: no objective or practice sections found")
            continue
        predictions = classifier.predict(text, k=len(HEADER) - 2)
        rows.append([assignment.name, assignment_id, *(format_prediction(*p) for p in predictions)])
    return rows
