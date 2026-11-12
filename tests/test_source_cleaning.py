"""Offline regression patterns observed in current official source extractions.

Publication furniture is quoted; article prose below is synthetic.
Real source IDs and inspection notes are recorded in reports/dataset_update_audit.md.
"""
import json
from io import BytesIO

import pytest

from src import ingest_corpus as ingest
from src.preprocess import clean_bitscan_pages, clean_wikinews_text
from src.source_extract import extract_html_text
from src.dataset import select_campus_body_page, normalize_document_length

PROSE = ("BITS Pilani students and faculty in Hyderabad and Goa reported research "
         "events. The article said that sources supported the findings.")


@pytest.mark.parametrize("footer", [
    "BITS Pilani, Pilani campus {page}",
    "{page} Issue 22 (Jan-June 2025), BITScan, Hyderabad Campus",
    "semester I notes 2021-22 bitscan, BITS Pilani K K Birla Goa Campus {page}",
    "Semester I 2015-16 bitscan, K.K. Birla Goa Campus {page}",
    "- {page} - Semester I, 2004-05",
    "\u2013 {page} \u2013 Semester I, 2004\u201305",
])
def test_observed_bitscan_numbered_furniture(footer):
    pages = [f"Article heading\n\n{PROSE}\n\n{footer.format(page=i)}" for i in range(1, 4)]
    cleaned = clean_bitscan_pages(pages)
    assert cleaned.count(PROSE) == 3
    for i in range(1, 4):
        assert footer.format(page=i) not in cleaned


def test_split_hyderabad_footer_and_meaningful_layout_like_mentions():
    footer = "Issue 22 (Jan-June 2025), BITScan, Hyderabad Campus"
    pages = [f"{PROSE}\n\n{i}\n{footer}" for i in (2, 3, 4)]
    cleaned = clean_bitscan_pages(pages)
    assert footer not in cleaned
    assert cleaned.count(PROSE) == 3
    # A single interior label and prose must not become a running footer.
    interior = "BITS Pilani, Pilani campus 1"
    page = "\n".join(["Heading", PROSE, PROSE, PROSE, interior, PROSE, PROSE, PROSE])
    assert interior in clean_bitscan_pages([page, page])
    single = f"{PROSE}\n{interior}"
    assert interior in clean_bitscan_pages([single])


@pytest.mark.parametrize("heading", [
    "Source", "Sources", "References", "Related news", "Related News", "Related stories",
    "Related wikinews",
    "Related", "External link", "External links", "Sister links", "See also",
])
def test_observed_wikinews_nonarticle_sections(heading):
    raw = (f"Tuesday, January 6, 2026\n\n{PROSE}\n\n== {heading} ==\n"
           "Citation or link list\n=== A subsection ===\nMore links\n"
           "== Interview ==\nThe researcher said the work will continue.")
    cleaned = clean_wikinews_text(raw)
    assert cleaned.startswith(PROSE)
    assert "Citation or link list" not in cleaned and "More links" not in cleaned
    assert "The researcher said" in cleaned


def test_wikinews_date_share_notice_and_prose_preservation():
    raw = ("Wednesday, February 1, 2023\n\n" + PROSE +
           "\nSources inside an article said Wikinews reported related news."
           "\nMonday, April 9, 2007 was the date discussed in the interview."
           "\nHave an opinion on this story? Share it!")
    cleaned = clean_wikinews_text(raw)
    assert "Wednesday, February" not in cleaned and "Have an opinion" not in cleaned
    assert PROSE in cleaned
    assert "Sources inside an article said Wikinews reported related news." in cleaned
    assert "Monday, April 9, 2007 was the date discussed" in cleaned


def test_observed_wikinews_sidebar_removed_but_article_lists_remain():
    html = (f"<div id='mw-content-text'><div class='infobox noprint desktop-only'>"
            "<ul><li>Unrelated current headlines</li></ul></div>"
            f"<p>{PROSE}</p><ul><li>Faculty explained a substantive research result.</li></ul>"
            "<h2>Source</h2><ul><li>Citation only</li></ul></div>").encode()
    cleaned = clean_wikinews_text(extract_html_text(html, wikinews=True))
    assert "Unrelated current headlines" not in cleaned and "Citation only" not in cleaned
    assert PROSE in cleaned and "substantive research result" in cleaned


@pytest.mark.parametrize("api_available", [True, False])
def test_api_and_html_paths_both_apply_wikinews_cleaning(monkeypatch, api_available):
    raw = "Tuesday, January 6, 2026\n\n" + PROSE + "\n\n== Sources ==\nCitation only"
    response = {"query": {"pages": {"1": {"extract": raw} if api_available else {}}}}
    monkeypatch.setattr(ingest.urllib.request, "urlopen",
                        lambda *a, **k: BytesIO(json.dumps(response).encode()))
    html = (f"<html><body><div id='mw-content-text'><p>Tuesday, January 6, 2026</p>"
            f"<p>{PROSE}</p><h2>Sources</h2><ul><li>Citation only</li></ul>"
            "<p>Have an opinion on this story? Share it!</p></div></body></html>").encode()
    monkeypatch.setattr(ingest, "download", lambda url: (html, "text/html", url))
    cleaned = ingest.fetch_wikinews_text("https://en.wikinews.org/wiki/Example")
    assert cleaned == PROSE


def test_campus_section_avoids_observed_front_matter():
    body = (PROSE + " ") * 12
    pages = ["Issue 22", "Editorial\n" + body,
             "To our alumni and students, we convey our gratitude.\n" + body,
             "Visit of the Vice-President of India\n\n" + body]
    chosen, page = select_campus_body_page("\f".join(pages))
    assert page == 4 and chosen.startswith("Visit of")
    model_text = normalize_document_length(chosen, 80)
    assert len(model_text.split()) <= 80 and model_text.endswith(".")
    assert model_text == normalize_document_length(select_campus_body_page("\f".join(pages))[0], 80)
    # Later content cannot affect the chosen section.
    assert select_campus_body_page("\f".join(pages) + "\f" + body)[0] == chosen


@pytest.mark.parametrize("opening", [
    "from the\neditor", "from the\neditor\nfrom the\neditor",
    "Contents", "Table of contents", "In Pursuit of Excellence",
    "A continuation from the adjacent column.\nfrom the\neditor\n\nDear Readers,",
    "Dear Readers,",
])
def test_campus_front_matter_patterns(opening):
    body = (PROSE + " ") * 8
    chosen, page = select_campus_body_page(opening + "\n" + body + "\fConvocation 2022\n" + body)
    assert page == 2 and chosen.startswith("Convocation")


def test_editorial_workshop_is_article_prose():
    body = ("An editorial workshop welcomed BITS Pilani faculty and students. " * 10)
    assert select_campus_body_page(body)[0] == body
