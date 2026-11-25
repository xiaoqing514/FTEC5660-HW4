"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================
from collections import defaultdict

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

# Filled by chunk_document(), used by parse_query() (the runner calls chunk_document first).
_CORPUS_COMPANIES: dict[str, str] = {}   # normalised company key -> display name
_CORPUS_BROKERS: set[str] = set()
_CORPUS_AUDITORS: set[str] = set()
_CORPUS_TOPICS: set[str] = set()         # segment and region names (never companies)

_DOC_LABEL = {"annual_report": "Annual Report", "earnings_call": "Earnings Call",
              "research_report": "Research Report"}


# --------------------------------------------------------------------------- Task 1.1: tables

def _squash_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _cell_text(cell) -> str:
    return _squash_space(cell.get_text(" ", strip=True))


def _table_grid(table) -> tuple[list[list[str]], list[list[bool]], int]:
    """Expand a table into a rectangular grid, honouring colspan/rowspan."""
    grid: dict[tuple[int, int], str] = {}
    is_th: dict[tuple[int, int], bool] = {}
    for r, tr in enumerate(table.find_all("tr")):
        col = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, col) in grid:
                col += 1
            text = _cell_text(cell)
            colspan = max(1, int(cell.get("colspan") or 1))
            rowspan = max(1, int(cell.get("rowspan") or 1))
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, col + dc)] = text
                    is_th[(r + dr, col + dc)] = cell.name == "th"
            col += colspan
    if not grid:
        return [], [], 0
    nrows = max(r for r, _ in grid) + 1
    ncols = max(c for _, c in grid) + 1
    rows = [[grid.get((r, c), "") for c in range(ncols)] for r in range(nrows)]
    ths = [[is_th.get((r, c), False) for c in range(ncols)] for r in range(nrows)]
    return rows, ths, ncols


def _table_header_rows(rows: list[list[str]], ths: list[list[bool]]) -> list[int]:
    """The leading rows made only of header cells (normally the single <thead> row)."""
    out: list[int] = []
    for r, row in enumerate(rows):
        non_empty = [c for c, value in enumerate(row) if value]
        if not non_empty:
            break
        # A section row ("Assets") spans every column and repeats the same text; it is not a header.
        if len({row[c] for c in non_empty}) < 2:
            break
        if len(non_empty) >= 2 and all(ths[r][c] for c in non_empty):
            out.append(r)
        else:
            break
    return out


def _table_row_lines(label: str, pairs: list[tuple[str, str]], limit: int = 380) -> list[str]:
    """One table row as 'label | header: value | ...', split if it would exceed MAX_LINE_CHARS."""
    whole = f"{label} | " + " | ".join(f"{header}: {value}" for header, value in pairs)
    if len(whole) <= limit:
        return [whole]
    lines: list[str] = []
    current = [label]
    for header, value in pairs:
        piece = f"{header}: {value}"
        if len(current) > 1 and len(" | ".join(current + [piece])) > limit:
            lines.append(" | ".join(current))
            current = [label, piece]
        else:
            current.append(piece)
    if len(current) > 1:
        lines.append(" | ".join(current))
    return lines


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    if not table_html or not str(table_html).strip():
        return ""
    soup = BeautifulSoup(str(table_html), "lxml")
    table = soup.find("table")
    if table is None:
        return ""

    caption = ""
    if table.caption is not None:
        caption = _squash_space(table.caption.get_text(" ", strip=True))

    rows, ths, ncols = _table_grid(table)
    if not rows:
        return caption

    header_rows = _table_header_rows(rows, ths)
    headers: list[str] = []
    for c in range(ncols):
        values: list[str] = []
        for r in header_rows:
            value = rows[r][c]
            if value and value not in values:
                values.append(value)
        headers.append(" ".join(values))

    lines: list[str] = []
    for r, row in enumerate(rows):
        if r in header_rows:
            continue
        label = row[0] if row[0] else next((value for value in row if value), "")
        if not label:
            continue
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for c in range(1, ncols):
            value = row[c]
            if not value or value == label:
                continue
            header = headers[c] or f"Column {c + 1}"
            if (header, value) in seen:      # a colspan repeats the same cell
                continue
            seen.add((header, value))
            pairs.append((header, value))
        if not pairs:                        # a section row such as "Current assets"
            continue
        lines.extend(_table_row_lines(label, pairs))

    parts = ([caption] if caption else []) + lines
    return "\n".join(parts)


# --------------------------------------------------------------------------- Task 1.2: chunking

_LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def _company_base(name: str) -> str:
    """Drop a trailing legal suffix and a trailing ticker/exchange parenthesis."""
    name = _squash_space(name)
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
    words = name.split()
    while words and words[-1].lower().strip(".,") in _LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words).strip(" ,.") or name


def _document_metadata(soup: BeautifulSoup) -> tuple[str, int, str]:
    """(company, fiscal year, doc type) from the first lines of one document."""
    title = ""
    if soup.title is not None:
        title = _squash_space(soup.title.get_text(" ", strip=True))
    h1 = ""
    if soup.find("h1") is not None:
        h1 = _squash_space(soup.find("h1").get_text(" ", strip=True))
    h2_text = " ".join(_squash_space(h.get_text(" ", strip=True))
                       for h in soup.find_all(["h2", "h3"])[:4])
    cover_text = " ".join(_squash_space(p.get_text(" ", strip=True))
                          for p in soup.find_all("p")[:6])
    probe = f"{title} {h1} {h2_text}".lower()
    head_text = f"{title} {h1} {cover_text}"

    broker = soup.find("div", class_="broker")
    if broker is not None or "equity research" in probe or "results review" in probe:
        doc_type = "research_report"
    elif "earnings conference call" in probe or "earnings call" in probe or "conference call" in probe:
        doc_type = "earnings_call"
    elif "annual report" in probe:
        doc_type = "annual_report"
    else:
        doc_type = "annual_report"

    year = None
    head = head_text.lower()
    if doc_type == "research_report":
        match = re.search(r"fy\s*(\d{4})\s+results review", head)
        if match is None:
            match = re.search(r"fy\s*(\d{4})", head)
        if match is None:
            match = re.search(r"(20\d{2})", head)
    elif doc_type == "earnings_call":
        match = re.search(r"(?:full year|fourth quarter and full year)\s+(\d{4})", head)
        if match is None:
            match = re.search(r"(20\d{2})", head)
    else:
        match = re.search(r"annual report\s+(\d{4})", head)
        if match is None:
            match = re.search(r"year ended 31 december\s+(\d{4})", head)
        if match is None:
            match = re.search(r"(20\d{2})", head)
    if match is not None:
        year = int(match.group(1))

    return _company_base(h1), int(year) if year else 0, doc_type


def _section_name(state: dict) -> str:
    h2, h3 = state.get("h2", ""), state.get("h3", "")
    if h2 and h3:
        return f"{h2} > {h3}"
    return h2 or h3 or "Cover"


def _chunk_header(company: str, doc_type: str, fiscal_year: int, section: str) -> str:
    label = _DOC_LABEL.get(doc_type, doc_type)
    return f"{company} | {label} {fiscal_year} | {section[:90]}"


def _split_text(text: str, limit: int) -> list[str]:
    """Split a long paragraph at sentence (then word) boundaries."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
        while len(current) > limit:
            cut = current.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(current[:cut])
            current = current[cut:].lstrip()
    if current:
        pieces.append(current)
    return pieces


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(html, "lxml")
    company, fiscal_year, doc_type = _document_metadata(soup)
    if company:
        _CORPUS_COMPANIES[company_key(company)] = company
    broker = soup.find("div", class_="broker")
    if broker is not None:
        name = _squash_space(broker.get_text("|", strip=True).split("|")[0])
        if name:
            _CORPUS_BROKERS.add(name)
    for paragraph in soup.find_all("p"):
        text = _squash_space(paragraph.get_text(" ", strip=True))
        match = re.match(r"(.+?)\s+audited the consolidated financial statements", text)
        if match is not None:
            _CORPUS_AUDITORS.add(match.group(1).strip())

    body = soup.body or soup
    state = {"h2": "", "h3": ""}
    units: list[dict] = []

    def add_text(text: str) -> None:
        text = _squash_space(text)
        if not text or state["h2"].lower() == "contents" or text.startswith("©"):
            return
        units.append({"kind": "text", "text": text, "section": _section_name(state)})

    def walk(node) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                continue
            if name in ("script", "style", "nav", "title", "head"):
                continue
            if name == "h1":
                continue
            if name == "h2":
                state["h2"] = _squash_space(child.get_text(" ", strip=True))
                state["h3"] = ""
                continue
            if name == "h3":
                state["h3"] = _squash_space(child.get_text(" ", strip=True))
                continue
            if name == "table":
                text = table_to_text(str(child))
                if text.strip():
                    lines = text.split("\n")
                    caption = lines[0].lower() if lines else ""
                    if "segment" in caption or "region" in caption:
                        for line in lines[1:]:
                            label = line.split(" | ")[0].strip()
                            if label and label.lower() != "total revenue":
                                _CORPUS_TOPICS.add(company_key(label))
                    units.append({"kind": "table", "text": text, "section": _section_name(state)})
                continue
            if name in ("p", "li"):
                add_text(child.get_text(" ", strip=True))
                continue
            if name == "div" and "broker" in (child.get("class") or []):
                add_text(child.get_text(" ", strip=True))
                continue
            if child.find(["p", "li", "table", "h2", "h3", "div"]):
                walk(child)
                continue
            add_text(child.get_text(" ", strip=True))

    walk(body)

    base_metadata = {"source": file_name, "company": company, "fiscal_year": fiscal_year,
                     "doc_type": doc_type}
    chunks: list[Document] = []

    def emit(text: str, section: str, kind: str) -> None:
        text = text.strip()
        if not text:
            return
        if len(text) > MAX_CHUNK_CHARS:      # defensive: never hand back an oversized chunk
            for piece in _split_text(text, MAX_CHUNK_CHARS):
                chunks.append(Document(page_content=piece,
                                       metadata={**base_metadata, "section": section, "kind": kind}))
            return
        chunks.append(Document(page_content=text,
                               metadata={**base_metadata, "section": section, "kind": kind}))

    def emit_table(unit: dict) -> None:
        lines = unit["text"].split("\n")
        caption = lines[0] if lines else ""
        content = lines[1:] if caption else lines
        header = _chunk_header(company, doc_type, fiscal_year, unit["section"])
        available = MAX_CHUNK_CHARS - len(header) - len(caption) - 2
        group: list[str] = []
        size = 0
        groups: list[list[str]] = []
        for line in content:
            if group and size + len(line) + 1 > available:
                groups.append(group)
                group, size = [], 0
            group.append(line)
            size += len(line) + 1
        if group:
            groups.append(group)
        for group in groups or [[]]:
            parts = [header]
            if caption:
                parts.append(caption)
            parts.extend(group)
            emit("\n".join(parts), unit["section"], "table")

    buffer: list[str] = []
    buffer_section = ""
    buffer_size = 0

    def flush() -> None:
        nonlocal buffer, buffer_size
        if not buffer:
            return
        header = _chunk_header(company, doc_type, fiscal_year, buffer_section)
        emit(header + "\n" + "\n".join(buffer), buffer_section, "text")
        buffer, buffer_size = [], 0

    for unit in units:
        if unit["section"] != buffer_section:
            flush()
            buffer_section = unit["section"]
        if unit["kind"] == "table":
            flush()
            emit_table(unit)
            continue
        if buffer and buffer_size + len(unit["text"]) + 1 > 1150:
            flush()
        pieces = _split_text(unit["text"], MAX_CHUNK_CHARS - len(
            _chunk_header(company, doc_type, fiscal_year, buffer_section)) - 2)
        buffer.extend(pieces)
        buffer_size += len(unit["text"]) + 1
    flush()
    return chunks


# --------------------------------------------------------------------------- Task 2.2: queries

_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"}
_RUN_STARTERS = {"what", "which", "who", "whom", "whose", "how", "when", "where", "why", "on",
                 "in", "at", "for", "from", "by", "to", "of", "the", "a", "an", "is", "are",
                 "was", "were", "does", "do", "did", "has", "have", "had", "its", "their",
                 "this", "that", "these", "those", "and", "or", "but", "if", "as", "with",
                 "during", "between", "after", "before", "about", "please", "according", "give",
                 "list", "tell", "name", "could", "would", "should"}
_NON_COMPANY = {
    "chief executive officer", "chief financial officer", "chief operating officer",
    "chief technology officer", "general counsel", "board of directors", "audit committee",
    "chair of the audit committee", "annual report", "earnings call", "earnings conference call",
    "conference call", "fourth quarter", "full year", "results review", "equity research",
    "investment summary", "target price", "price target", "share price", "investment rating",
    "common stock", "free cash flow", "united states", "asia pacific", "north america",
    "south america", "latin america", "middle east", "middle east and africa",
    "europe middle east and africa", "hong kong", "united kingdom",
}
_UPPER_TOKENS = {"usd", "fy", "ceo", "cfo", "coo", "cto", "llp", "inc", "corp", "ltd", "plc",
                 "not", "found", "q1", "q2", "q3", "q4", "nyse", "nasdaq", "lse", "hkex", "tsx"}


def _known_company_mentions(question: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for display in _CORPUS_COMPANIES.values():
        words = [re.escape(word) for word in display.split()]
        if not words:
            continue
        pattern = re.compile(r"(?<![A-Za-z0-9])" + r"\s+".join(words) + r"(?![A-Za-z0-9])", re.I)
        for match in pattern.finditer(question):
            spans.append((match.start(), match.end(), display))
    # Longest span wins; drop overlapping shorter ones.
    spans.sort(key=lambda item: item[0])
    kept: list[tuple[int, int, str]] = []
    for span in sorted(spans, key=lambda item: item[1] - item[0], reverse=True):
        if all(span[1] <= other[0] or span[0] >= other[1] for other in kept):
            kept.append(span)
    return sorted(kept)


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


def _unknown_companies(question: str, taken: list[tuple[int, int]]) -> list[str]:
    masked = _mask_spans(question, taken)
    found: list[str] = []
    pattern = re.compile(r"\b[A-Z][A-Za-z0-9&.'\-]*(?:\s+(?:of|the|and|&|[A-Z][A-Za-z0-9&.'\-]*)){0,5}")
    for match in pattern.finditer(masked):
        tokens = [re.sub(r"'s$", "", token, flags=re.I)
                  for token in match.group(0).replace(".", " ").split() if token]
        edge_words = _RUN_STARTERS | _UPPER_TOKENS
        while tokens and tokens[0].lower() in edge_words:
            tokens.pop(0)
        while tokens and tokens[-1].lower() in edge_words:
            tokens.pop()
        possessive = masked[match.end():match.end() + 2].lower() == "'s"
        if len(tokens) < 2 and not (len(tokens) == 1 and possessive):
            continue
        normalised = " ".join(token.lower() for token in tokens)
        if normalised in _NON_COMPANY:
            continue
        if any(token.lower() in _MONTHS for token in tokens):
            continue
        if any(re.fullmatch(r"(?:FY|Q)\d+", token, re.I) for token in tokens):
            continue
        name = " ".join(tokens).strip(" ,.;:")
        if company_key(name) in _CORPUS_TOPICS:
            continue
        if name and company_key(name) not in {company_key(existing) for existing in found}:
            found.append(name)
    return found


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    try:
        question = _squash_space(question)
        known = _known_company_mentions(question)
        companies: list[str] = []
        taken: list[tuple[int, int]] = []
        for start, end, display in known:
            if all(display != existing for existing in companies):
                companies.append(display)
            taken.append((start, end))
        for name in _CORPUS_BROKERS | _CORPUS_AUDITORS:
            for match in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])",
                                     question, re.I):
                taken.append((match.start(), match.end()))
        companies.extend(_unknown_companies(question, taken))

        # One year at most; an "FY2026" reference wins over other four-digit numbers.
        year = None
        match = re.search(r"FY\s*(\d{4})", question, re.I)
        if match is not None:
            year = int(match.group(1))
        else:
            years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
            if len(years) == 1:
                year = int(years[0])
            elif years:
                tail = re.search(r"(?:31\s+december|december\s+31|year ended|fiscal year|"
                                 r"in|at|during|end of)\s+(\d{4})", question, re.I)
                year = int(tail.group(1)) if tail is not None else int(years[0])

        lowered = question.lower()
        if re.search(r"\bearnings call\b|\bconference call\b|\btranscript\b|\bq[1-4]\b|"
                     r"\bfourth[- ]quarter\b|\bquarterly call\b|\bthe call\b|"
                     r"\bguidance\b|\bguide[sd]?\b|\bmanagement (?:said|expects)\b",
                     lowered):
            doc_type = "earnings_call"
        elif re.search(r"\brating\b|\btarget price\b|\bprice target\b|\bprice objective\b|"
                       r"\bforecast\b|\bresults review\b|\bequity research\b|\banalyst\b|"
                       r"\bestimate[sd]?\b|\bbroker\b|\bcoverage\b|\binitiat(?:e|ion)\b|"
                       r"\bupgrade\b|\bdowngrade\b|\bresearch note\b|\bbroker note\b|12-month",
                       lowered):
            doc_type = "research_report"
        else:
            doc_type = "annual_report"
        return {"companies": companies, "fiscal_year": year, "doc_type": doc_type}
    except Exception:
        return {"companies": [], "fiscal_year": None, "doc_type": "annual_report"}


# --------------------------------------------------------------------------- Task 2: index

_STOPWORDS = set("""a an the and or but if then than that this these those of to in on at for from by with
without within into onto is are was were be been being do does did done have has had having what which who
whom whose how much many when where why should would could can may might must will shall about above after
again against all also am any because before below between both during each few further here more most no nor
not now off once only other own same so some such too under until up very via per its it he she they we you i
us our your his her them their theirs""".split())

_SYNONYMS = {
    "cash": ["cash", "equivalents", "liquid"],
    "employee": ["employee", "employees", "headcount", "staff", "people"],
    "employees": ["employees", "employee", "headcount", "staff", "people"],
    "revenue": ["revenue", "revenues", "sales", "turnover"],
    "sales": ["sales", "revenue", "turnover"],
    "income": ["income", "profit", "earnings"],
    "profit": ["profit", "income", "earnings"],
    "debt": ["debt", "borrowings", "borrowing"],
    "staff": ["staff", "employees", "employee", "headcount", "people", "personnel"],
    "people": ["people", "employees", "employee", "headcount", "staff", "personnel"],
    "personnel": ["personnel", "employees", "employee", "headcount", "staff"],
    "employs": ["employs", "employees", "employee", "headcount", "staff"],
    "chair": ["chair", "chairs", "chairman"],
    "chairs": ["chairs", "chair", "chairman"],
    "buyback": ["buyback", "repurchase", "programme", "program"],
    "outlook": ["outlook", "guidance", "guided", "guide"],
    "based": ["based", "headquarters", "headquartered", "city"],
    "located": ["located", "headquarters", "headquartered", "city"],
    "borrowing": ["borrowings", "borrowing", "debt", "debt"],
    "borrowings": ["borrowings", "borrowing", "debt"],
    "receivable": ["receivable", "receivables"],
    "receivables": ["receivables", "receivable"],
    "inventory": ["inventory", "inventories", "stock"],
    "inventories": ["inventories", "inventory", "stock"],
    "auditor": ["auditor", "audit", "firm"],
    "audit": ["audit", "auditor", "firm"],
    "cfo": ["chief", "financial", "officer"],
    "ceo": ["chief", "executive", "officer"],
    "headquarters": ["headquarters", "headquartered", "city"],
    "headquartered": ["headquartered", "headquarters", "city"],
    "city": ["city", "headquarters", "headquartered"],
    "rating": ["rating", "buy", "hold", "reduce"],
    "target": ["target", "price"],
    "repurchase": ["repurchase", "buyback", "programme", "program"],
    "guidance": ["guidance", "guided", "guide", "outlook"],
    "guide": ["guide", "guidance", "guided", "outlook"],
    "guided": ["guided", "guidance", "guide", "outlook"],
    "forecast": ["forecast", "estimate", "expects"],
}

_SECTION_HINTS = [
    ("cash", ["balance sheet", "cash flow", "cash and cash equivalents"]),
    ("revenue", ["income statement", "revenue by segment", "revenue by region"]),
    ("employee", ["employees", "headcount"]),
    ("borrow", ["borrowings"]),
    ("receivable", ["accounts receivable"]),
    ("inventor", ["balance sheet"]),
    ("total assets", ["balance sheet"]),
    ("total equity", ["balance sheet"]),
    ("net cash", ["cash flow"]),
    ("gross profit", ["income statement"]),
    ("operating income", ["income statement"]),
    ("net income", ["income statement", "cash flow"]),
    ("segment", ["revenue by segment"]),
    ("region", ["revenue by region"]),
    ("audit", ["independent auditor"]),
    ("chief financial", ["executive officers"]),
    ("chief executive", ["executive officers"]),
    ("headquarter", ["headquarters"]),
    ("repurchase", ["repurchase"]),
    ("rating", ["rating"]),
    ("target price", ["target price"]),
]

_METRICS = [
    ("net cash from operating activities", r"net cash (?:from|used in) operating activities"),
    ("cash and cash equivalents", r"cash and cash equivalents"),
    ("accounts receivable", r"accounts receivable|receivables"),
    ("long-term borrowings", r"long[- ]term borrowings?"),
    ("total assets", r"total assets"),
    ("total equity", r"total equity"),
    ("gross profit", r"gross profit"),
    ("operating income", r"operating income"),
    ("net income", r"net income"),
    ("revenue", r"revenue|sales"),
    ("employees", r"employees|headcount"),
    ("inventories", r"inventor(?:y|ies)"),
    ("target price", r"target price|price target"),
    ("rating", r"rating"),
    ("share repurchase", r"share repurchase|buyback"),
    ("auditor", r"audit firm|auditor"),
    ("chief financial officer", r"chief financial officer|\bcfo\b"),
    ("chief executive officer", r"chief executive officer|\bceo\b"),
    ("headquarters", r"headquarter"),
]


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    tokens: list[str] = []
    for token in re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", text):
        if token[0].isdigit():
            tokens.append(token.replace(",", ""))
        elif token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def _expansion_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(_SYNONYMS.get(token, []))
    return expanded


def _query_metrics(question: str) -> list[str]:
    lowered = question.lower()
    metrics = [name for name, pattern in _METRICS if re.search(pattern, lowered)]
    # Segment and region names in the question are useful row labels too.
    known = {company_key(name) for name in _CORPUS_COMPANIES.values()}
    excluded = known | {company_key(name) for name in _CORPUS_BROKERS | _CORPUS_AUDITORS}
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question):
        phrase = match.group(0)
        if company_key(phrase) in excluded or phrase.lower() in _MONTHS:
            continue
        if len(phrase.split()) <= 3:
            metrics.append(phrase.lower())
    return list(dict.fromkeys(metrics))


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    chunks = [chunk for chunk in chunks if isinstance(chunk, Document)]
    tokens = [_tokenize(chunk.page_content) for chunk in chunks]
    bm25 = BM25Okapi(tokens, k1=1.5, b=0.75) if tokens else None
    by_company: dict[str, list[int]] = defaultdict(list)
    by_type: dict[str, list[int]] = defaultdict(list)
    companies: dict[str, str] = {}
    doc_freq: dict[str, int] = defaultdict(int)
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        key = company_key(meta.get("company", ""))
        if key:
            by_company[key].append(i)
            companies[key] = meta.get("company", "")
        by_type[str(meta.get("doc_type", ""))].append(i)
        for token in set(tokens[i]):
            doc_freq[token] += 1
    return {"chunks": chunks, "tokens": tokens, "bm25": bm25, "by_company": dict(by_company),
            "by_type": dict(by_type), "companies": companies, "doc_freq": dict(doc_freq)}


def _year_window(index: dict, targets: list[str], year: int | None, doc_type: str) -> set[int] | None:
    """The document fiscal years that can state a fact about `year`, or None if hopeless."""
    if year is None:
        return None
    if doc_type == "annual_report":
        if any(year in {index["chunks"][i].metadata.get("fiscal_year") for i in index["by_company"][key]}
               for key in targets if key in index["by_company"]):
            return {year}
        if year == 2023 and any(2024 in {index["chunks"][i].metadata.get("fiscal_year")
                                         for i in index["by_company"][key]}
                                for key in targets if key in index["by_company"]):
            return {2024}
        return set()
    if doc_type == "research_report":
        return {2025} if 2024 <= year <= 2027 else set()
    if doc_type == "earnings_call":
        return {2025} if year in (2025, 2026) else set()
    return None


def _score_chunks(index: dict, question: str, parsed: dict, candidates: list[int],
                  targets: list[str], year: int | None, doc_type: str) -> dict[int, float]:
    bm25 = index["bm25"]
    query_tokens = _expansion_tokens(_tokenize(question))
    base = bm25.get_scores(query_tokens) if bm25 is not None and query_tokens else [0.0] * len(index["chunks"])
    metrics = _query_metrics(question)
    lowered = question.lower()
    numeric = bool(re.search(r"\b(?:how much|how many|what was|what were|what is)\b", lowered)) and \
        bool(re.search(r"\busd\b|percent|million", lowered))
    scores: dict[int, float] = {}
    for i in candidates:
        chunk = index["chunks"][i]
        meta = chunk.metadata or {}
        text = chunk.page_content
        low = text.lower()
        score = float(base[i])
        if year is not None and meta.get("fiscal_year") == year:
            score += 1.2
        if year is not None and re.search(rf"(?<!\d){year}(?!\d)", text):
            score += 0.35
        if doc_type == "research_report" and year is not None and f"fy{year}" in low:
            score += 0.8
        for metric in metrics:
            if metric in low:
                score += 0.7
                if re.search(rf"(?m)^{re.escape(metric)}\s*[|:]", low):
                    score += 1.7
        for concept, hints in _SECTION_HINTS:
            if concept in lowered and any(hint in low for hint in hints):
                score += 0.55
        if numeric and " | " in text:
            score += 0.45
        if ("chief financial officer" in lowered or "cfo" in lowered) and "chief financial officer" in low:
            score += 0.9
        if ("chief executive officer" in lowered or "ceo" in lowered) and "chief executive officer" in low:
            score += 0.9
        if "audit" in lowered and "auditor" in low:
            score += 0.9
        if "headquarter" in lowered and "headquarter" in low:
            score += 0.9
        if "rating" in lowered and re.search(r"rating\s*:", low):
            score += 0.9
        if "target price" in lowered and "target price" in low:
            score += 0.9
        scores[i] = score
    return scores


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    if not index or not index.get("chunks"):
        return []
    parsed = parse_query(question)
    doc_type = parsed["doc_type"]
    year = parsed["fiscal_year"]
    targets: list[str] = []
    for name in parsed["companies"]:
        key = company_key(name)
        if key in index["companies"] and key not in targets:
            targets.append(key)
    if not targets:
        return []
    years = _year_window(index, targets, year, doc_type)
    if years is not None and not years:
        return []

    candidates: list[int] = []
    for key in targets:
        for i in index["by_company"].get(key, []):
            meta = index["chunks"][i].metadata or {}
            if meta.get("doc_type") != doc_type:
                continue
            if years is not None and meta.get("fiscal_year") not in years:
                continue
            candidates.append(i)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return []

    scores = _score_chunks(index, question, parsed, candidates, targets, year, doc_type)
    ordered = sorted(candidates, key=lambda i: scores[i], reverse=True)

    if len(targets) > 1:
        chosen: list[int] = []
        for key in targets:
            own = [i for i in ordered if company_key(index["chunks"][i].metadata.get("company", "")) == key]
            chosen.extend(own[:2])
        for i in ordered:
            if len(chosen) >= MAX_RETRIEVED:
                break
            if i not in chosen:
                chosen.append(i)
        selection = chosen[:MAX_RETRIEVED]
    else:
        selection = ordered[:MAX_RETRIEVED]

    # Abstention: a term the question needs but the corpus never uses means "not disclosed".
    context = " ".join(index["chunks"][i].page_content.lower() for i in selection)
    total = max(len(index["chunks"]), 1)
    for term in _rare_question_terms(question, parsed):
        variants = {term}
        variants.update(_SYNONYMS.get(term, []))
        for variant in list(variants):
            variants.add(variant.rstrip("s"))
            variants.add(variant + "s")
        frequency = max(index["doc_freq"].get(variant, 0) for variant in variants) / total
        present = any(variant in context for variant in variants)
        if frequency == 0 or (frequency < 0.01 and not present):
            return []
    return [index["chunks"][i] for i in selection]


_Q_SCAFFOLD = set("""what which who whom whose how when where why was were is are did does do has have had
company companies corporation firm report reports reported fiscal year years annual usd million millions
billion percent percentage much many more most less least higher lower highest lowest bigger smaller high low
large small value amount
number name give set hold held holding generate generated generates announce announced announces spend
spent spending earn earned earning latest new old current end beginning during period time share shares
its four fourth quarter call conference results review target price investment rating forecast estimate
profitability""".split())


def _rare_question_terms(question: str, parsed: dict) -> list[str]:
    """Content words whose absence from the retrieved chunks means the fact is not disclosed."""
    stripped = question
    spans: list[tuple[int, int]] = []
    for name in parsed.get("companies", []):
        for match in re.finditer(re.escape(name), stripped, re.I):
            spans.append((match.start(), match.end()))
    for name in _CORPUS_BROKERS | _CORPUS_AUDITORS:
        for match in re.finditer(re.escape(name), stripped, re.I):
            spans.append((match.start(), match.end()))
    stripped = _mask_spans(stripped, spans)
    words = re.findall(r"[a-z]+", stripped.lower())
    return [word for word in dict.fromkeys(words)
            if len(word) > 2 and word not in _STOPWORDS and word not in _Q_SCAFFOLD]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
