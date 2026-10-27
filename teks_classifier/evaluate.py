"""Measure the model against the TEKS teachers cited in their own lesson plans.

Many lesson plans list the standards they cover in a "TEKS" section. The model
never sees that section (see lesson_parser.INPUT_HEADINGS), so it works as an
answer key for real-world accuracy.

    python -m teks_classifier.evaluate                # lessons from the Canvas course in .env
    python -m teks_classifier.evaluate lessons.json   # or from a saved file

lessons.json maps assignment IDs to {"name": ..., "description": <lesson HTML>}.
"""

import argparse
import json
import os
from math import comb

from canvasapi import Canvas
from dotenv import load_dotenv

from .classifier import TeksClassifier
from .config import DEFAULT_MODEL, Config
from .lesson_parser import lesson_text, teks_codes


def load_lessons(path: str | None) -> list[tuple[str, str | None]]:
    if path:
        with open(path, encoding="utf-8") as file:
            return [(lesson["name"], lesson.get("description")) for lesson in json.load(file).values()]
    config = Config.from_env(require_sheets=False)
    course = Canvas(config.canvas_url, config.canvas_token).get_course(config.course_id)
    return [(assignment.name, assignment.description) for assignment in course.get_assignments()]


def chance_of_hit(cited: int, labels: int, guesses: int) -> float:
    """Probability that `guesses` random distinct labels include at least one cited one."""
    return 1 - comb(labels - cited, guesses) / comb(labels, guesses)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lessons", nargs="?", help="JSON file of lessons (default: fetch from Canvas)")
    parser.add_argument("--model", help=f"model ID or local path (default: $TEKS_MODEL or {DEFAULT_MODEL})")
    args = parser.parse_args()

    load_dotenv()
    classifier = TeksClassifier(args.model or os.environ.get("TEKS_MODEL", DEFAULT_MODEL))
    known_labels = classifier.labels

    top1 = top3 = evaluated = 0
    chance1 = chance3 = 0.0
    for name, description in load_lessons(args.lessons):
        cited = teks_codes(description)
        text = lesson_text(description)
        if not cited or not text:
            continue

        predicted = [label for label, _ in classifier.predict(text, k=3)]
        evaluated += 1
        top1 += predicted[0] in cited
        top3 += any(label in cited for label in predicted)
        predictable = len(cited & known_labels)
        chance1 += chance_of_hit(predictable, len(known_labels), 1)
        chance3 += chance_of_hit(predictable, len(known_labels), 3)

        mark = "✓" if predicted[0] in cited else "~" if set(predicted) & cited else "✗"
        print(f"{mark} {name[:50]:<50}  cited: {' '.join(sorted(cited)):<20}  predicted: {' '.join(predicted)}")

    if not evaluated:
        print("No lessons had both a TEKS section and lesson text to classify.")
        return
    print(f"\n{evaluated} lessons with teacher-cited TEKS")
    print(f"Top-1 accuracy: {top1}/{evaluated} ({top1 / evaluated:.0%}), random guessing: {chance1 / evaluated:.0%}")
    print(f"Top-3 accuracy: {top3}/{evaluated} ({top3 / evaluated:.0%}), random guessing: {chance3 / evaluated:.0%}")


if __name__ == "__main__":
    main()
