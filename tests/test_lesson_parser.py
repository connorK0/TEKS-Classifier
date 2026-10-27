from teks_classifier.lesson_parser import lesson_text, split_sections, teks_codes

INLINE_HEADINGS = """
<p><strong>TEKS</strong>:&nbsp; History 3C, 12A</p>
<p><strong>ESSENTIAL QUESTION</strong>: Why did settlers move west?</p>
<p><strong>DO NOW</strong>: Discuss the painting with a partner.</p>
<p><strong>GUIDED PRACTICE</strong>:</p>
<p>Read the article on <strong>homesteading</strong> and discuss it as a class.</p>
<p><strong>INDEPENDENT PRACTICE</strong>:</p>
<p>Write a paragraph about the Homestead Act.</p>
"""

LIST_OF_STANDARDS = """
<h4><strong>TEKS:</strong></h4>
<ul>
  <li><strong>USH.3A Analyze</strong> political issues such as <strong>Indian policies</strong></li>
  <li><strong>USH.25B Describe</strong> the Americanization movement</li>
</ul>
<h4><strong>INSTRUCTIONAL OBJECTIVE:</strong></h4>
<p><em>Students will be able to describe how westward expansion affected Native Americans.</em></p>
"""


def test_splits_on_bold_headings():
    sections = split_sections(INLINE_HEADINGS)
    assert list(sections) == ["teks", "essential question", "do now", "guided practice", "independent practice"]
    assert sections["essential question"] == "Why did settlers move west?"


def test_bold_words_inside_a_section_do_not_end_it():
    assert split_sections(INLINE_HEADINGS)["guided practice"] == (
        "Read the article on homesteading and discuss it as a class."
    )


def test_lesson_text_keeps_objectives_and_practice_only():
    text = lesson_text(INLINE_HEADINGS)
    assert "Why did settlers move west?" in text
    assert "Homestead Act" in text
    assert "painting" not in text  # DO NOW is logistics
    assert "12A" not in text  # the answer key must never reach the model


def test_teks_codes_from_inline_list():
    assert teks_codes(INLINE_HEADINGS) == {"3.C", "12.A"}


def test_teks_codes_from_bulleted_standards():
    assert teks_codes(LIST_OF_STANDARDS) == {"3.A", "25.B"}
    assert lesson_text(LIST_OF_STANDARDS) == (
        "Students will be able to describe how westward expansion affected Native Americans."
    )


def test_teks_codes_ignore_years_in_standard_text():
    html = "<p><strong>TEKS -</strong></p><p>USH.16A - Analyze economic growth in the 1920s</p>"
    assert teks_codes(html) == {"16.A"}


def test_empty_or_missing_descriptions():
    for html in (None, "", "<p>Unit test on Friday.</p>"):
        assert lesson_text(html) == ""
        assert teks_codes(html) == set()
