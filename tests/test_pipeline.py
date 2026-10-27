from types import SimpleNamespace

from teks_classifier.pipeline import build_rows, format_prediction

LESSON = "<p><strong>LEARNING OBJECTIVE</strong>: Explain the causes of the Great Depression.</p>"


class FakeClassifier:
    def __init__(self):
        self.seen = []

    def predict(self, text, k=3):
        self.seen.append(text)
        return [("16.A", 0.612), ("16.B", 0.2), ("17.A", 0.049)][:k]


def assignment(id, description=LESSON, name="Lesson"):
    return SimpleNamespace(id=id, name=name, description=description)


def test_builds_one_row_per_new_assignment():
    rows = build_rows([assignment(101, name="4.1 - The Crash")], FakeClassifier())
    assert rows == [["4.1 - The Crash", "101", "16.A (61%)", "16.B (20%)", "17.A (5%)"]]


def test_skips_ids_already_in_the_sheet_even_though_canvas_ids_are_ints():
    classifier = FakeClassifier()
    rows = build_rows([assignment(101), assignment(102)], classifier, skip_ids={"101"})
    assert [row[1] for row in rows] == ["102"]
    assert len(classifier.seen) == 1


def test_skips_assignments_with_no_lesson_text():
    classifier = FakeClassifier()
    rows = build_rows([assignment(101, description=None), assignment(102, description="<p>Unit test</p>")], classifier)
    assert rows == []
    assert classifier.seen == []


def test_classifier_sees_lesson_text_not_html():
    classifier = FakeClassifier()
    build_rows([assignment(101)], classifier)
    assert classifier.seen == ["Explain the causes of the Great Depression."]


def test_format_prediction_rounds_to_whole_percent():
    assert format_prediction("1.A", 0.4449) == "1.A (44%)"
