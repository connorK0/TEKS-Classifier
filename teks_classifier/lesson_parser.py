"""Pull the useful text out of a Canvas lesson-plan description.

Lesson plans are written in Canvas's rich-text editor, so they arrive as HTML.
Teachers mark each part of the plan with a bold heading ("LEARNING OBJECTIVE:",
"GUIDED PRACTICE", ...) followed by one or more paragraphs. This module splits
the HTML on those headings and keeps only the parts that describe *what* is
being taught, which is what the classifier needs.
"""

import re

from bs4 import BeautifulSoup, NavigableString, Tag

# Headings whose text is passed to the classifier, after normalization.
# Logistics such as "DO NOW" or links to slide decks are left out on purpose.
INPUT_HEADINGS = {
    "learning objective",
    "instructional objective",
    "essential question",
    "guided practice",
    "group practice",
    "self guided practice",
    "independent practice",
    "independent work",
    "individual practice",
}

TEKS_HEADING = "teks"

# A heading only counts if it starts a paragraph or heading tag. Bold words in
# the middle of a sentence, or inside list items, are emphasis, not headings.
_BLOCK_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6"]
_MAX_HEADING_WORDS = 5

# Matches TEKS citations as teachers write them: "12A", "12.A", "USH.12A".
_TEKS_CODE = re.compile(r"(?<!\d)(\d{1,2})\.?([A-Z])(?![A-Za-z])")


def _normalize_heading(text: str) -> str:
    return " ".join(text.lower().strip(" :-–—").split())


def _is_heading(tag: Tag) -> bool:
    text = tag.get_text(" ", strip=True)
    name = _normalize_heading(text)
    if not name or any(ch.isdigit() for ch in name) or len(name.split()) > _MAX_HEADING_WORDS:
        return False
    block = tag.find_parent(_BLOCK_TAGS)
    return block is not None and block.get_text(" ", strip=True).startswith(text)


def split_sections(html: str | None) -> dict[str, str]:
    """Return {normalized heading: section text} in document order."""
    if not html:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    headings = {id(tag) for tag in soup.find_all(["strong", "b"]) if _is_heading(tag)}

    sections: dict[str, list[str]] = {}
    current = None
    for node in soup.descendants:
        if isinstance(node, Tag) and id(node) in headings:
            current = _normalize_heading(node.get_text(" ", strip=True))
            sections.setdefault(current, [])
        elif (
            type(node) is NavigableString
            and current is not None
            and not any(id(parent) in headings for parent in node.parents)
        ):
            sections[current].append(node)

    return {
        name: " ".join(" ".join(parts).split()).lstrip(" :-–—")
        for name, parts in sections.items()
    }


def lesson_text(html: str | None) -> str:
    """The text of a lesson plan that the classifier should see."""
    sections = split_sections(html)
    return " ".join(text for name, text in sections.items() if name in INPUT_HEADINGS and text)


def teks_codes(html: str | None) -> set[str]:
    """TEKS codes the teacher cited in the lesson's own "TEKS" section, e.g. {"3.C", "12.A"}."""
    cited = split_sections(html).get(TEKS_HEADING, "")
    return {f"{number}.{letter}" for number, letter in _TEKS_CODE.findall(cited)}
