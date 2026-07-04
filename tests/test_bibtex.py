from __future__ import annotations

from arxiv_recommender import bibtex


class TestExtractArxivId:
    def test_new_style_id_from_url(self):
        entry = {"url": "http://arxiv.org/abs/2210.02747"}
        assert bibtex.extract_arxiv_id(entry) == "2210.02747"

    def test_new_style_id_5_digit_from_url(self):
        entry = {"url": "http://arxiv.org/abs/2210.12345"}
        assert bibtex.extract_arxiv_id(entry) == "2210.12345"

    def test_strips_version_suffix(self):
        entry = {"url": "http://arxiv.org/abs/2210.02747v3"}
        assert bibtex.extract_arxiv_id(entry) == "2210.02747"

    def test_old_style_id_from_url(self):
        entry = {"url": "http://arxiv.org/abs/hep-th/9901001"}
        assert bibtex.extract_arxiv_id(entry) == "hep-th/9901001"

    def test_doi_field(self):
        entry = {"doi": "10.48550/arXiv.2210.02747"}
        assert bibtex.extract_arxiv_id(entry) == "2210.02747"

    def test_doi_field_case_insensitive(self):
        entry = {"doi": "10.48550/ARXIV.2210.02747"}
        assert bibtex.extract_arxiv_id(entry) == "2210.02747"

    def test_note_field(self):
        entry = {"note": "arXiv:2210.02747"}
        assert bibtex.extract_arxiv_id(entry) == "2210.02747"

    def test_note_field_with_space(self):
        entry = {"note": "arXiv: 2210.02747"}
        assert bibtex.extract_arxiv_id(entry) == "2210.02747"

    def test_prefers_url_over_doi_over_note(self):
        entry = {
            "url": "http://arxiv.org/abs/1111.11111",
            "doi": "10.48550/arXiv.2222.22222",
            "note": "arXiv:3333.33333",
        }
        assert bibtex.extract_arxiv_id(entry) == "1111.11111"

    def test_falls_back_to_doi_when_no_url(self):
        entry = {"doi": "10.48550/arXiv.2222.22222", "note": "arXiv:3333.33333"}
        assert bibtex.extract_arxiv_id(entry) == "2222.22222"

    def test_no_arxiv_id_returns_none(self):
        entry = {"url": "http://example.com/paper", "title": "Some Paper"}
        assert bibtex.extract_arxiv_id(entry) is None

    def test_empty_entry_returns_none(self):
        assert bibtex.extract_arxiv_id({}) is None


class TestClean:
    def test_strips_braces(self):
        assert bibtex._clean("{Attention} is All You Need") == "Attention is All You Need"

    def test_collapses_whitespace(self):
        assert bibtex._clean("foo   bar\nbaz") == "foo bar baz"

    def test_none_passthrough(self):
        assert bibtex._clean(None) is None


class TestParseAuthors:
    def test_splits_on_and(self):
        result = bibtex._parse_authors("Smith, John and Doe, Jane")
        assert result == ["Smith, John", "Doe, Jane"]

    def test_single_author(self):
        assert bibtex._parse_authors("Smith, John") == ["Smith, John"]

    def test_none_returns_empty(self):
        assert bibtex._parse_authors(None) == []

    def test_empty_string_returns_empty(self):
        assert bibtex._parse_authors("") == []


class TestPublishedDate:
    def test_year_and_month_name(self):
        entry = {"year": "2022", "month": "oct"}
        assert bibtex._published_date(entry) == "2022-10"

    def test_year_and_numeric_month(self):
        entry = {"year": "2022", "month": "3"}
        assert bibtex._published_date(entry) == "2022-03"

    def test_year_only(self):
        entry = {"year": "2022"}
        assert bibtex._published_date(entry) == "2022"

    def test_no_year_returns_none(self):
        assert bibtex._published_date({}) is None

    def test_unrecognized_month_falls_back_to_year(self):
        entry = {"year": "2022", "month": "notamonth"}
        assert bibtex._published_date(entry) == "2022"


class TestParseLibrary:
    def test_parses_entries_and_skips_missing_ids(self, tmp_path):
        bib_content = """
@article{smith2022,
  title = {A Great Paper},
  author = {Smith, John and Doe, Jane},
  year = {2022},
  month = {oct},
  url = {http://arxiv.org/abs/2210.02747},
  keywords = {cs.LG, cs.AI},
}
@article{norefpaper,
  title = {No ArXiv Reference},
  author = {Someone, Else},
  year = {2020},
}
"""
        bib_path = tmp_path / "library.bib"
        bib_path.write_text(bib_content, encoding="utf-8")

        papers, skipped = bibtex.parse_library(bib_path)

        assert len(papers) == 1
        assert skipped == ["norefpaper"]
        paper = papers[0]
        assert paper["arxiv_id"] == "2210.02747"
        assert paper["title"] == "A Great Paper"
        assert paper["authors"] == ["Smith, John", "Doe, Jane"]
        assert paper["categories"] == ["cs.LG", "cs.AI"]
        assert paper["published_date"] == "2022-10"
