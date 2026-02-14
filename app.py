# app.py
import re
import time
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Iterable
from collections import Counter, defaultdict
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from tenacity import retry, stop_after_attempt, wait_exponential

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# =============================
# Optional dependencies
# =============================
try:
    import docx  # python-docx module
except Exception:
    docx = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# For Word export (DOCX)
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None


# =============================
# Usage counter system
# =============================
USAGE_FILE = "usage_stats.json"


def _default_usage():
    return {
        "app_runs": 0,
        "files_processed": 0,
        "citations_checked": 0,
        "references_checked": 0,
    }


def load_usage_stats():
    if not os.path.exists(USAGE_FILE):
        return _default_usage()
    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure keys exist
        base = _default_usage()
        base.update({k: int(data.get(k, base[k])) for k in base.keys()})
        return base
    except Exception:
        return _default_usage()


def save_usage_stats(stats):
    try:
        with open(USAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass


def increment_usage(app_run=False, file=False, cites=0, refs=0):
    stats = load_usage_stats()
    if app_run:
        stats["app_runs"] += 1
    if file:
        stats["files_processed"] += 1
    stats["citations_checked"] += int(cites or 0)
    stats["references_checked"] += int(refs or 0)
    save_usage_stats(stats)
    return stats


# =============================
# Constants
# =============================
YEAR = r"(?:19|20)\d{2}[a-z]?"  # allows 2020a
CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"

REF_HEADINGS = [
    r"^\s*references?\s*(?:list)?\s*[:|]?\s*$",
    r"^\s*bibliograph(?:y|ies)\s*[:|]?\s*$",
    r"^\s*works\s+cited\s*[:|]?\s*$",
    r"^\s*literature\s+cited\s*[:|]?\s*$",
]

ORG_ALIASES = {
    "unctad": ["unctad", "united nations conference on trade and development"],
    "who": ["who", "world health organization", "world health organisation"],
    "oecd": [
        "oecd",
        "organisation for economic co-operation and development",
        "organization for economic cooperation and development",
    ],
    "world bank": ["world bank", "international bank for reconstruction and development", "ibrd"],
    "world bank group": ["world bank group"],
    "imf": ["imf", "international monetary fund"],
    "un": ["un", "united nations"],
    "unesco": ["unesco"],
    "unicef": ["unicef"],
}

ORG_ACRONYMS = {
    "UNCTAD", "WHO", "OECD", "IMF", "UN", "UNESCO", "UNICEF",
    "WORLD BANK", "WORLD BANK GROUP", "IBRD"
}

NONNAME_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by",
    "from", "into", "at", "than", "that", "this", "these", "those",
    "recent", "minimum", "maximum", "mathematical", "basis", "adopting", "adopt", "model",
    "framework", "table", "figure", "chapter", "section", "appendix", "equation",
    "definition", "method", "methods", "result", "results", "discussion",
    "similarly", "moreover", "however", "therefore", "thus", "also", "further", "additionally",
    "consequently", "notably", "specifically", "generally", "overall", "first", "second", "finally",
    "public", "construct", "africa", "europe", "america"
}


# =============================
# Data classes
# =============================
@dataclass
class InTextCitation:
    style: str
    raw: str
    key: str
    pretty: str
    author_or_org: Optional[str] = None
    year: Optional[str] = None
    number: Optional[int] = None


@dataclass
class ReferenceEntry:
    raw: str
    key: str
    pretty: str
    author_or_org: Optional[str] = None
    year: Optional[str] = None
    number: Optional[int] = None


# =============================
# Normalisation helpers
# =============================
def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_token(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9\- ']", "", s)
    return norm_spaces(s)


def titleish(s: str) -> str:
    if not s:
        return s
    if s.strip().upper() in ORG_ACRONYMS:
        return s.strip().upper()
    words = s.strip().split()
    return " ".join((w[:1].upper() + w[1:]) if w else w for w in words)


def canon_org(name: str) -> str:
    n = norm_token(name)
    for canon, variants in ORG_ALIASES.items():
        for v in variants:
            if n == norm_token(v):
                return canon
    return n


def key_author_year(author_surname: str, year: str) -> str:
    return f"au_{norm_token(author_surname)}_{year.lower()}"


def key_org_year(org: str, year: str) -> str:
    return f"org_{canon_org(org)}_{year.lower()}"


def key_numeric(n: int) -> str:
    return f"n_{n}"


def is_known_org(text: str) -> bool:
    t = text.strip()
    if t.upper() in ORG_ACRONYMS:
        return True
    c = canon_org(t)
    return c in ORG_ALIASES.keys()


def looks_like_two_authors(name: str) -> bool:
    parts = name.strip().split()
    return len(parts) == 2 and all(p[:1].isupper() for p in parts)


def _first_surname_from_author_blob(blob: str) -> Optional[str]:
    if not blob:
        return None
    b = blob.strip()
    b = re.sub(r"^(see|e\.g\.|cf\.)\s+", "", b, flags=re.IGNORECASE).strip()
    b = re.sub(r"\s+et\s+al\.?$", "", b, flags=re.IGNORECASE).strip()
    first = b.split(",")[0].strip()
    first = re.split(r"\s+(?:and|&)\s+", first, flags=re.IGNORECASE)[0].strip()
    m = re.search(r"([A-Z][A-Za-z\-']+)", first)
    return m.group(1) if m else None


def looks_like_person_surname(token: str) -> bool:
    if not token:
        return False
    t = token.strip()

    # reject possessive like Adam's / Adam’s
    if re.search(r"(?:'s|’s)$", t, flags=re.IGNORECASE):
        return False

    if not re.fullmatch(r"[A-Z][A-Za-z\-']{1,40}", t):
        return False

    if norm_token(t) in NONNAME_STOPWORDS:
        return False

    return True


def format_author_blob_for_display(authors_blob: str) -> str:
    if not authors_blob:
        return ""
    s = authors_blob.strip()
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"\s+and\s+", " and ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================
# DOCX block iterator (keeps order)
# =============================
def iter_block_items(document) -> Iterable[Tuple[str, str]]:
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent = document.element.body
    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            p = Paragraph(child, document)
            t = p.text.strip()
            if t:
                yield ("p", t)
        elif isinstance(child, CT_Tbl):
            tbl = Table(child, document)
            for row in tbl.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    yield ("t", " ".join(cells))


# =============================
# File readers
# =============================
def read_docx(file) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(file)
    parts = []
    for _, txt in iter_block_items(d):
        parts.append(txt)
    return "\n".join(parts)


def read_pdf(file) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed")
    out = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            out.append(p.extract_text() or "")
    return "\n".join(out)


# =============================
# Reference section detection
# =============================
def split_by_heading(text: str) -> Tuple[str, str, str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        for pat in REF_HEADINGS:
            if re.search(pat, s, flags=re.IGNORECASE):
                main = "\n".join(lines[:i]).strip()
                refs = "\n".join(lines[i + 1:]).strip()
                return main, refs, f"Found heading: {s}"
    return text.strip(), "", "No heading found"


def ref_line_score(line: str) -> int:
    s = line.strip()
    if not s:
        return 0
    if re.match(r"^\s*(\[\d+\]|\d+[\.\)])\s+", s):
        return 5
    if re.match(rf"^[A-Z][A-Za-z\-']+\s*,.*\(\s*{YEAR}\s*\)", s):
        return 6
    if re.match(rf"^[A-Z][A-Za-z\-']+\s*,.*\b{YEAR}\b", s):
        return 4
    if re.match(rf"^[A-Z][A-Z&\- ]{{2,}}\.\s*\(\s*{YEAR}\s*\)", s):
        return 6
    if re.match(rf"^[A-Z][A-Za-z&.\- ]{{2,}}?\s*\(\s*{YEAR}\s*\)", s):
        return 3
    return 0


def auto_detect_references_start(text: str) -> Tuple[int, float]:
    lines = text.splitlines()
    if len(lines) < 40:
        idx = max(0, int(len(lines) * 0.70))
        return idx, 0.2

    window = 40
    start = int(len(lines) * 0.35)
    scores = []
    for i in range(start, max(start + 1, len(lines) - window)):
        score = sum(ref_line_score(lines[j]) for j in range(i, i + window))
        scores.append((i, score))

    if not scores:
        idx = max(0, int(len(lines) * 0.70))
        return idx, 0.2

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    best_i, best_s = scores_sorted[0]
    second_s = scores_sorted[1][1] if len(scores_sorted) > 1 else 0

    if best_s <= 0:
        return best_i, 0.0

    margin = best_s - second_s
    conf = max(0.0, min(1.0, (best_s / (best_s + 25)) + (margin / (best_s + 1)) * 0.35))
    return best_i, conf


# =============================
# In-text citation extraction (author-year)
# =============================
def extract_author_year_citations(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []

    narr_people = re.compile(
        rf"""
        \b
        (?P<authors>
            [A-Z][A-Za-z\-']+
            (?:\s*,\s*[A-Z][A-Za-z\-']+)*
            (?:\s*,?\s*(?:and|&)\s*[A-Z][A-Za-z\-']+)?
            |
            [A-Z][A-Za-z\-']+\s+et\s+al\.
        )
        \s*
        \(\s*(?P<year>{YEAR})\s*\)
        """,
        flags=re.VERBOSE
    )

    for m in narr_people.finditer(text):
        authors_blob = m.group("authors").strip()
        y = m.group("year")

        authors_blob_clean = re.sub(r"\s+et\s+al\.\s*$", "", authors_blob, flags=re.IGNORECASE).strip()
        first = _first_surname_from_author_blob(authors_blob_clean) or authors_blob_clean.split(",")[0].strip()

        if not looks_like_person_surname(first):
            continue

        disp_authors = format_author_blob_for_display(authors_blob)
        key = key_author_year(first, y)
        pretty = f"{disp_authors} ({y})"
        out.append(InTextCitation("author-year", m.group(0), key, pretty, first, y))

    narr_org = re.compile(rf"\b([A-Z][A-Za-z&.\- ]{{2,}}?)\s*\(\s*({YEAR})\s*\)")
    for m in narr_org.finditer(text):
        org, y = m.group(1).strip(), m.group(2)
        if looks_like_two_authors(org):
            continue
        if is_known_org(org):
            key = key_org_year(org, y)
            pretty = f"{titleish(org)} ({y})"
            out.append(InTextCitation("org-year", m.group(0), key, pretty, org, y))

    paren_block = re.compile(rf"\(([^()]*?\b{YEAR}\b[^()]*)\)")
    for m in paren_block.finditer(text):
        block = m.group(1)
        parts = [p.strip() for p in block.split(";") if p.strip()]

        for p in parts:
            yrm = re.search(rf"\b({YEAR})\b", p)
            if not yrm:
                continue
            y = yrm.group(1)

            left = p
            m_left = re.search(rf"^(.+?)\s*,\s*{YEAR}\b", p)
            if m_left:
                left = m_left.group(1).strip()

            if is_known_org(left) and not looks_like_two_authors(left):
                key = key_org_year(left, y)
                pretty = f"({p})"
                out.append(InTextCitation("org-year", f"({p})", key, pretty, left, y))
                continue

            first_author = _first_surname_from_author_blob(left) or _first_surname_from_author_blob(p)
            if not first_author:
                continue
            if not looks_like_person_surname(first_author):
                continue

            key = key_author_year(first_author, y)
            pretty = f"({p})"
            out.append(InTextCitation("author-year", f"({p})", key, pretty, first_author, y))

    return out


# =============================
# Numeric citation extraction
# =============================
def extract_ieee_numeric_citations(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []
    pat = re.compile(r"\[(\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*\]")
    for m in pat.finditer(text):
        raw = m.group(0)
        inside = m.group(1)
        chunks = [c.strip() for c in inside.split(",")]
        nums: List[int] = []
        for c in chunks:
            r = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", c)
            if r:
                a, b = int(r.group(1)), int(r.group(2))
                if a <= b and (b - a) <= 2000:
                    nums.extend(range(a, b + 1))
            elif c.isdigit():
                nums.append(int(c))
        for n in nums:
            out.append(InTextCitation("numeric", raw, key_numeric(n), f"[{n}]", number=n))
    return out


def extract_vancouver_parentheses_numeric(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []
    pat = re.compile(r"\((\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*\)")
    for m in pat.finditer(text):
        raw = m.group(0)
        inside = m.group(1)

        # exclude (2020) which is likely author-year
        if re.fullmatch(rf"\s*{YEAR}\s*", inside):
            continue

        chunks = [c.strip() for c in inside.split(",")]
        nums: List[int] = []
        for c in chunks:
            r = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", c)
            if r:
                a, b = int(r.group(1)), int(r.group(2))
                if a <= b and (b - a) <= 2000:
                    nums.extend(range(a, b + 1))
            elif c.isdigit():
                nums.append(int(c))
        for n in nums:
            out.append(InTextCitation("numeric", raw, key_numeric(n), f"({n})", number=n))
    return out


# =============================
# Reference entry extraction
# =============================
def split_reference_entries(ref_text: str) -> List[str]:
    ref_text = ref_text.strip()
    if not ref_text:
        return []

    lines = [l.rstrip() for l in ref_text.splitlines() if l.strip()]
    entries: List[str] = []
    buf = ""

    start_pat = re.compile(
        r"^\s*(\[\d+\]|\d+[\.\)])\s+"
        r"|^\s*[A-Z][A-Za-z\-']+\s*,"
        r"|^\s*[A-Z][A-Z&\- ]{2,}\."
        r"|^\s*(World Bank|World Bank Group|UNCTAD|OECD|WHO|IMF|UNESCO|UNICEF)\b",
        flags=re.IGNORECASE
    )

    for line in lines:
        if start_pat.search(line) and buf:
            entries.append(buf.strip())
            buf = line.strip()
        else:
            buf = (buf + " " + line.strip()).strip() if buf else line.strip()

    if buf:
        entries.append(buf.strip())

    if len(entries) == 1:
        big = entries[0]
        chunks = re.split(r"(?=(?:\s|^)(?:\[\d+\]|\d+[\.\)])\s+)", big)
        chunks = [norm_spaces(c) for c in chunks if norm_spaces(c)]
        if len(chunks) >= 2:
            entries = chunks

    return entries


def _strip_leading_numbering(entry: str) -> str:
    return re.sub(r"^\s*(\[\d+\]|\d+[\.\)])\s+", "", entry).strip()


def parse_reference_author_year(entry: str) -> Optional[ReferenceEntry]:
    e = _strip_leading_numbering(entry)
    if not e:
        return None

    org_m = re.search(rf"^([A-Z][A-Z&\- ]{{2,}})\.\s*\(\s*({YEAR})\s*\)", e)
    if org_m:
        org, y = org_m.group(1).strip(), org_m.group(2)
        if is_known_org(org):
            key = key_org_year(org, y)
            pretty = f"{titleish(org)} ({y})"
            return ReferenceEntry(entry, key, pretty, org, y)

    org2 = re.search(rf"^([A-Z][A-Za-z&.\- ]{{2,}}?)\.?\s*\(\s*({YEAR})\s*\)", e)
    if org2:
        org, y = org2.group(1).strip(), org2.group(2)
        if is_known_org(org) and not looks_like_two_authors(org):
            key = key_org_year(org, y)
            pretty = f"{titleish(org)} ({y})"
            return ReferenceEntry(entry, key, pretty, org, y)

    m = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\(\s*({YEAR})\s*\)", e)
    if m:
        au, y = m.group(1), m.group(2)
        key = key_author_year(au, y)
        pretty = f"{titleish(au)} ({y})"
        return ReferenceEntry(entry, key, pretty, au, y)

    m2 = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\b({YEAR})\b", e)
    if m2:
        au, y = m2.group(1), m2.group(2)
        key = key_author_year(au, y)
        pretty = f"{titleish(au)} ({y})"
        return ReferenceEntry(entry, key, pretty, au, y)

    return None


def parse_reference_numeric(entry: str) -> Optional[ReferenceEntry]:
    e = entry.strip()

    m = re.match(r"^\s*\[(\d+)\]\s*(.+)$", e)
    if m:
        n = int(m.group(1))
        return ReferenceEntry(entry, key_numeric(n), f"[{n}]", number=n)

    m = re.match(r"^\s*(\d+)[\.\)]\s+(.+)$", e)
    if m:
        n = int(m.group(1))
        return ReferenceEntry(entry, key_numeric(n), f"[{n}]", number=n)

    return None


# =============================
# Reference-style detection + reformat + DOCX export
# =============================
STYLE_APA = "APA-like"
STYLE_HARVARD = "Harvard-like"
STYLE_NUMERIC = "Numeric"


def detect_reference_style(ref_entries: List[str]) -> Tuple[str, Dict[str, int]]:
    counts = {STYLE_APA: 0, STYLE_HARVARD: 0, STYLE_NUMERIC: 0}

    for raw in ref_entries:
        r = raw.strip()
        if not r:
            continue

        if re.match(r"^\s*(\[\d+\]|\d+[\.\)])\s+", r):
            counts[STYLE_NUMERIC] += 1
            continue

        if re.search(rf"^[A-Z][A-Za-z\-']+\s*,.+?\(\s*{YEAR}\s*\)\.", r):
            counts[STYLE_APA] += 1
            continue

        if re.search(rf"^[A-Z][A-Za-z\-']+\s*,.+?\b{YEAR}\b", r) and "(" not in r[:60]:
            counts[STYLE_HARVARD] += 1
            continue

    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    return dominant, counts


def _get_first_author_and_year_from_ref(ref_raw: str) -> Tuple[str, str]:
    e = _strip_leading_numbering(ref_raw)

    m_org = re.match(r"^\s*([A-Z][A-Za-z&.\- ]{2,}?)\s*[\.\,]?\s*\(?\s*(" + YEAR + r")\s*\)?", e)
    if m_org:
        who = m_org.group(1).strip()
        y = m_org.group(2).strip()
        return titleish(who), y

    m_au = re.match(r"^\s*([A-Z][A-Za-z\-']+)\s*,.*?\(?\s*(" + YEAR + r")\s*\)?", e)
    if m_au:
        return titleish(m_au.group(1).strip()), m_au.group(2).strip()

    y = ""
    ym = re.search(rf"\b({YEAR})\b", e)
    if ym:
        y = ym.group(1)
    return "Unknown", y


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "citation-crosscheck/1.4"})
    return s


SESSION = build_session()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=7))
def _get_json(url: str, params: dict) -> dict:
    r = SESSION.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def extract_doi(text: str) -> Optional[str]:
    if not text:
        return None

    m = re.search(r"https?://doi\.org/(10\.\d{4,9}/[^\s<>\"]+)", text, flags=re.IGNORECASE)
    if m:
        doi = m.group(1)
    else:
        m2 = re.search(r"(10\.\d{4,9}/[^\s<>\"]+)", text, flags=re.IGNORECASE)
        if not m2:
            return None
        doi = m2.group(1)

    doi = doi.strip().strip(").,;:]}>\"'")
    doi = re.sub(r"&quot;|&gt;|&lt;|&amp;", "", doi)
    doi = doi.strip().strip(").,;:]}>\"'")
    return doi if doi.lower().startswith("10.") else None


@st.cache_data(show_spinner=False)
def crossref_lookup_by_doi(doi: str) -> Optional[dict]:
    try:
        data = _get_json(f"{CROSSREF_API}/{doi}", params={})
        return data.get("message")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def crossref_search(query: str, rows: int = 10) -> List[dict]:
    data = _get_json(CROSSREF_API, params={"query.bibliographic": query, "rows": rows})
    return data.get("message", {}).get("items", []) or []


@st.cache_data(show_spinner=False)
def openalex_search(query: str, per_page: int = 10) -> List[dict]:
    data = _get_json(OPENALEX_API, params={"search": query, "per-page": per_page})
    return data.get("results", []) or []


def crossref_item_year(it: dict) -> Optional[int]:
    issued = it.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        try:
            return int(issued[0][0])
        except Exception:
            pass
    published_print = it.get("published-print", {}).get("date-parts", [])
    if published_print and published_print[0]:
        try:
            return int(published_print[0][0])
        except Exception:
            pass
    created = it.get("created", {}).get("date-parts", [])
    if created and created[0]:
        try:
            return int(created[0][0])
        except Exception:
            pass
    return None


def crossref_first_author_family(it: dict) -> str:
    authors = it.get("author") or []
    fam = authors[0].get("family") if authors else ""
    return fam or ""


def crossref_title(it: dict) -> str:
    t = it.get("title") or []
    return (t[0] if t else "") or ""


def openalex_year(it: dict) -> Optional[int]:
    y = it.get("publication_year")
    try:
        return int(y) if y else None
    except Exception:
        return None


def openalex_first_author_family(it: dict) -> str:
    authorships = it.get("authorships") or []
    if not authorships:
        return ""
    dn = (authorships[0].get("author") or {}).get("display_name") or ""
    return (dn.split()[-1] if dn else "") or ""


def openalex_title(it: dict) -> str:
    return (it.get("title") or "") or ""


def openalex_doi(it: dict) -> str:
    ids = it.get("ids") or {}
    d = ids.get("doi") or ""
    if d:
        d = d.replace("https://doi.org/", "").strip()
        d = d.strip(").,;:]}>\"'")
    return d


def _as_int_year(y: Optional[str]) -> Optional[int]:
    if not y:
        return None
    m = re.search(r"(19|20)\d{2}", str(y))
    return int(m.group(0)) if m else None


def author_match_ok(ref_surname: str, cand_surname: str) -> bool:
    if not ref_surname or not cand_surname:
        return False
    return norm_token(ref_surname) == norm_token(cand_surname)


def year_match_ok(ref_year: Optional[int], cand_year: Optional[int]) -> bool:
    if ref_year is None or cand_year is None:
        return False
    if ref_year == cand_year:
        return True
    return abs(ref_year - cand_year) == 1


def guess_title_snippet(ref: str) -> str:
    r = _strip_leading_numbering(ref)
    t = re.sub(rf"\(.*?{YEAR}.*?\)", " ", r)
    t = re.sub(r"^[^\.]{1,220}\.\s*", " ", t)
    t = norm_spaces(t)
    words = t.split()
    return " ".join(words[:22])[:260]


def verify_reference_online(
    ref_obj: ReferenceEntry,
    throttle_s: float = 0.2,
    use_crossref: bool = True,
    use_openalex: bool = True
) -> dict:
    ref_entry = ref_obj.raw or ""
    doi = extract_doi(ref_entry)

    ref_year = _as_int_year(ref_obj.year)
    who = (ref_obj.author_or_org or "").strip()
    who_is_org = ref_obj.key.startswith("org_")
    ref_surname = who if (who and (not who_is_org) and looks_like_person_surname(who)) else ""

    title_snip = guess_title_snippet(ref_entry)

    # DOI first
    if doi and use_crossref:
        cr = crossref_lookup_by_doi(doi)
        if cr:
            y = crossref_item_year(cr)
            fam = crossref_first_author_family(cr)
            title = crossref_title(cr)
            return {
                "status": "verified",
                "source": "crossref_doi",
                "score": 100,
                "doi": doi,
                "matched_year": str(y or ""),
                "matched_first_author": fam,
                "matched_title": (title or "")[:180],
                "query_used": "doi_lookup",
                "error_crossref": "",
                "error_openalex": "",
            }
        return {
            "status": "not_found",
            "source": "crossref_doi",
            "score": 0,
            "doi": doi,
            "matched_year": "",
            "matched_first_author": "",
            "matched_title": "",
            "query_used": "doi_lookup",
            "error_crossref": "",
            "error_openalex": "",
        }

    parts = []
    if who:
        parts.append(who)
    if ref_year:
        parts.append(str(ref_year))
    if title_snip:
        parts.append(title_snip)

    query = " ".join(parts).strip() or _strip_leading_numbering(ref_entry)[:200]
    time.sleep(max(0.0, throttle_s))

    best = {
        "status": "not_found",
        "source": "",
        "score": 0,
        "doi": "",
        "matched_year": "",
        "matched_first_author": "",
        "matched_title": "",
        "query_used": query[:220],
        "error_crossref": "",
        "error_openalex": "",
    }

    if use_crossref:
        try:
            items = crossref_search(query, rows=10)
            for it in items:
                cand_year = crossref_item_year(it)
                cand_fam = crossref_first_author_family(it)
                cand_title = crossref_title(it)
                cand_doi = (it.get("DOI") or "").strip()

                if not who_is_org:
                    if ref_surname and not author_match_ok(ref_surname, cand_fam):
                        continue
                    if ref_year and not year_match_ok(ref_year, cand_year):
                        continue
                else:
                    if ref_year and not year_match_ok(ref_year, cand_year):
                        continue

                score = fuzz.WRatio(title_snip, cand_title) if (title_snip and cand_title) else 70
                status = "verified" if score >= 90 else ("likely" if score >= 82 else "needs_review")

                if score > best["score"]:
                    best = {
                        "status": status,
                        "source": "crossref",
                        "score": int(score),
                        "doi": cand_doi,
                        "matched_year": str(cand_year or ""),
                        "matched_first_author": cand_fam,
                        "matched_title": (cand_title or "")[:180],
                        "query_used": query[:220],
                        "error_crossref": "",
                        "error_openalex": best.get("error_openalex", ""),
                    }
        except Exception as e:
            best["error_crossref"] = str(e)[:220]

    if use_openalex:
        try:
            items = openalex_search(query, per_page=10)
            for it in items:
                cand_year = openalex_year(it)
                cand_fam = openalex_first_author_family(it)
                cand_title = openalex_title(it)
                cand_doi = openalex_doi(it)

                if not who_is_org:
                    if ref_surname and not author_match_ok(ref_surname, cand_fam):
                        continue
                    if ref_year and not year_match_ok(ref_year, cand_year):
                        continue
                else:
                    if ref_year and not year_match_ok(ref_year, cand_year):
                        continue

                score = fuzz.WRatio(title_snip, cand_title) if (title_snip and cand_title) else 70
                status = "verified" if score >= 90 else ("likely" if score >= 82 else "needs_review")

                if score > best["score"]:
                    best = {
                        "status": status,
                        "source": "openalex",
                        "score": int(score),
                        "doi": cand_doi,
                        "matched_year": str(cand_year or ""),
                        "matched_first_author": cand_fam,
                        "matched_title": (cand_title or "")[:180],
                        "query_used": query[:220],
                        "error_crossref": best.get("error_crossref", ""),
                        "error_openalex": "",
                    }
        except Exception as e:
            best["error_openalex"] = str(e)[:220]

    return best


def format_reference_entry(ref_raw: str, target_style: str, idx: int = 1, enrich: bool = False) -> str:
    base = _strip_leading_numbering(ref_raw).strip()

    if enrich:
        doi = extract_doi(base)
        if doi:
            cr = crossref_lookup_by_doi(doi)
            if cr:
                title = (cr.get("title") or [""])[0] if cr.get("title") else ""
                authors = cr.get("author") or []
                year = crossref_item_year(cr)
                container = (cr.get("container-title") or [""])[0] if cr.get("container-title") else ""
                volume = cr.get("volume") or ""
                issue = cr.get("issue") or ""
                page = cr.get("page") or ""

                def _fmt_author(a: dict) -> str:
                    fam = a.get("family") or ""
                    giv = a.get("given") or ""
                    initials = ""
                    for part in giv.replace(".", " ").split():
                        if part:
                            initials += part[0].upper() + ". "
                    initials = initials.strip()
                    if fam and initials:
                        return f"{fam}, {initials}"
                    return fam or giv or ""

                author_str = ", ".join([_fmt_author(a) for a in authors if _fmt_author(a)]) if authors else ""
                year_str = str(year) if year else ""

                if target_style == STYLE_APA:
                    vol_issue = volume
                    if issue:
                        vol_issue = f"{volume}({issue})" if volume else f"({issue})"
                    tail = ", ".join([p for p in [container, vol_issue] if p]).strip()
                    pieces = [
                        p for p in [
                            author_str,
                            f"({year_str})." if year_str else "",
                            f"{title}." if title else "",
                            tail,
                            page
                        ] if p
                    ]
                    out = " ".join(pieces).strip()
                    out = f"{out} https://doi.org/{doi}".strip()
                    return out

                if target_style == STYLE_HARVARD:
                    head = author_str
                    if year_str:
                        head = f"{head}, {year_str}."
                    vol_issue = volume
                    if issue:
                        vol_issue = f"{volume}({issue})" if volume else f"({issue})"
                    mid = " ".join([p for p in [f"{title}." if title else "", container] if p]).strip()
                    tail = " ".join([p for p in [vol_issue, (page + ".") if page else ""] if p]).strip()
                    out = " ".join([p for p in [head, mid, tail] if p]).strip()
                    out = f"{out} https://doi.org/{doi}".strip()
                    return out

                if target_style == STYLE_NUMERIC:
                    vol_issue = volume
                    if issue:
                        vol_issue = f"{volume}({issue})" if volume else f"({issue})"
                    pp = f"pp. {page}" if page else ""
                    out = f"[{idx}] {author_str}, \"{title},\" {container}, {vol_issue}, {pp}, {year_str}."
                    out = f"{out} https://doi.org/{doi}".strip()
                    return norm_spaces(out)

    who, y = _get_first_author_and_year_from_ref(base)

    if target_style == STYLE_APA:
        if re.search(rf"\(\s*{YEAR}\s*\)", base):
            return base
        if y:
            return re.sub(rf"\b{re.escape(y)}\b", f"({y}).", base, count=1)

    if target_style == STYLE_HARVARD:
        if re.search(rf"\(\s*{YEAR}\s*\)", base):
            return re.sub(rf"\(\s*({YEAR})\s*\)", r"\1", base)
        return base

    if target_style == STYLE_NUMERIC:
        if re.match(r"^\s*\[\d+\]", ref_raw) or re.match(r"^\s*\d+[\.\)]", ref_raw):
            return ref_raw
        return f"[{idx}] {base}"

    return base


def export_references_to_docx(formatted_refs: List[str], title: str = "References") -> bytes:
    if DocxDocument is None:
        raise RuntimeError("python-docx not available for DOCX export")

    doc = DocxDocument()
    doc.add_heading(title, level=1)
    for r in formatted_refs:
        doc.add_paragraph(r)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


# =============================
# Year mismatch check
# =============================
def base_key(k: str) -> str:
    parts = k.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 3 else k


def year_from_key(k: str) -> Optional[str]:
    parts = k.split("_")
    return parts[-1] if len(parts) >= 3 else None


def find_year_mismatches(cite_keys: List[str], ref_keys: List[str]) -> pd.DataFrame:
    cite_map = defaultdict(set)
    ref_map = defaultdict(set)

    for ck in cite_keys:
        b, y = base_key(ck), year_from_key(ck)
        if y:
            cite_map[b].add(y)

    for rk in ref_keys:
        b, y = base_key(rk), year_from_key(rk)
        if y:
            ref_map[b].add(y)

    rows = []
    for b in sorted(set(cite_map.keys()) & set(ref_map.keys())):
        if cite_map[b] != ref_map[b]:
            rows.append({
                "author_or_org_key": b,
                "years_in_text": ", ".join(sorted(cite_map[b])),
                "years_in_references": ", ".join(sorted(ref_map[b])),
            })
    return pd.DataFrame(rows)


# =============================
# PDF report
# =============================
def make_pdf_report(style_name: str,
                    summary: dict,
                    df_missing: pd.DataFrame,
                    df_uncited: pd.DataFrame,
                    df_mismatch: pd.DataFrame,
                    df_verify: pd.DataFrame) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, y, "Citation Crosscheck Report")
    y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Style: {style_name}")
    y -= 0.8 * cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Summary")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        c.drawString(2 * cm, y, f"{k.replace('_',' ').title()}: {v}")
        y -= 0.5 * cm
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm

    def draw_table(title: str, df: pd.DataFrame, max_rows: int = 30):
        nonlocal y
        c.showPage()
        y = height - 2 * cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, y, title)
        y -= 0.7 * cm
        c.setFont("Helvetica", 9)

        if df is None or df.empty:
            c.drawString(2 * cm, y, "None detected.")
            return

        cols = list(df.columns)
        c.drawString(2 * cm, y, " | ".join(cols))
        y -= 0.4 * cm
        c.line(2 * cm, y, width - 2 * cm, y)
        y -= 0.3 * cm

        for _, row in df.head(max_rows).iterrows():
            line = " | ".join(str(row[col])[:72] for col in cols)
            c.drawString(2 * cm, y, line)
            y -= 0.45 * cm
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 9)

        if len(df) > max_rows:
            c.drawString(2 * cm, y, f"... showing {max_rows} of {len(df)} rows")
            y -= 0.6 * cm

    draw_table("Missing in References", df_missing)
    draw_table("Uncited References", df_uncited)
    draw_table("Year Mismatches", df_mismatch)
    draw_table("Online Verification", df_verify)

    c.save()
    return buf.getvalue()


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Citation Crosschecker", layout="wide")

st.markdown(
    """
<div style="margin-top:6px; margin-bottom:10px; padding:10px 12px; border:1px solid #e6e6e6; border-radius:10px;">
  <div style="font-size:13px; font-weight:600;">
    © Prof. Anokye M. Adam
  </div>
  <div style="font-size:12px; margin-top:6px; line-height:1.35;">
    Disclaimer: This tool can make mistakes, including missed or incorrect citation matches and online metadata errors.
    Always verify results against your manuscript, style guide, and original sources before submission.
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.title("Citation Crosschecker")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload DOCX, PDF, or TXT", type=["docx", "pdf", "txt"])
with col2:
    pasted = st.text_area("Or paste your manuscript text", height=170)

style = st.selectbox(
    "Citation style to check",
    ["APA/Harvard (author–year)", "IEEE (numeric [1])", "Vancouver (numeric (1))"],
)

text = ""
if uploaded is not None:
    name = uploaded.name.lower()
    try:
        if name.endswith(".docx"):
            text = read_docx(uploaded)
        elif name.endswith(".pdf"):
            text = read_pdf(uploaded)
        elif name.endswith(".txt"):
            text = uploaded.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Could not read file: {e}")
elif pasted.strip():
    text = pasted

if not text.strip():
    st.info("Upload a file or paste text to begin.")
    st.stop()

increment_usage(app_run=True)

main_text, ref_text, ref_msg = split_by_heading(text)
auto_idx, auto_conf = auto_detect_references_start(text) if ref_msg == "No heading found" else (None, 1.0)

st.subheader("References detection")
st.write(ref_msg)

force_manual = (ref_msg == "No heading found") and (auto_conf < 0.60)
manual = st.checkbox(
    "Manually choose where References start (recommended if heading not found)",
    value=force_manual or (ref_msg == "No heading found"),
)

if manual:
    lines = text.splitlines()
    guess = auto_idx if auto_idx is not None else max(0, int(len(lines) * 0.70))
    idx = st.slider("Select the line where the References section starts", 0, max(0, len(lines) - 1), guess)
    main_text = "\n".join(lines[:idx]).strip()
    ref_text = "\n".join(lines[idx:]).strip()

st.caption(f"Main text length: {len(main_text):,} chars | References length: {len(ref_text):,} chars")

with st.expander("Preview detected References section"):
    st.text(ref_text[:8000] if ref_text else "No references detected yet")

if not ref_text.strip():
    st.warning("References section is empty. Paste your References or adjust the slider.")
    ref_text = st.text_area("Paste References section here", height=240)

# Extract citations and references
if style.startswith("APA/Harvard"):
    cites = extract_author_year_citations(main_text)
    ref_raw = split_reference_entries(ref_text)
    refs = [parse_reference_author_year(r) for r in ref_raw]
    refs = [r for r in refs if r is not None]
elif style.startswith("IEEE"):
    cites = extract_ieee_numeric_citations(main_text)
    ref_raw = split_reference_entries(ref_text)
    refs = [parse_reference_numeric(r) for r in ref_raw]
    refs = [r for r in refs if r is not None]
else:
    cites = extract_vancouver_parentheses_numeric(main_text)
    ref_raw = split_reference_entries(ref_text)
    refs = [parse_reference_numeric(r) for r in ref_raw]
    refs = [r for r in refs if r is not None]

# Record usage counts
increment_usage(file=True, cites=len(cites), refs=len(refs))

st.divider()
st.subheader("Reference style check and reformat (export to Word)")

det_style, det_counts = detect_reference_style(ref_raw)

st.write(
    f"Detected dominant reference style: **{det_style}** "
    f"(APA-like: {det_counts[STYLE_APA]}, Harvard-like: {det_counts[STYLE_HARVARD]}, Numeric: {det_counts[STYLE_NUMERIC]})"
)

target_style = st.selectbox(
    "Reformat all reference entries into",
    [STYLE_APA, STYLE_HARVARD, STYLE_NUMERIC],
    index=[STYLE_APA, STYLE_HARVARD, STYLE_NUMERIC].index(det_style) if det_style in [STYLE_APA, STYLE_HARVARD, STYLE_NUMERIC] else 0
)

enrich_with_doi = st.checkbox(
    "Improve formatting using DOI metadata (Crossref lookup when DOI exists)",
    value=False
)

if st.button("Generate reformatted References list"):
    formatted = []
    for i, raw in enumerate(ref_raw, start=1):
        formatted.append(format_reference_entry(raw, target_style, idx=i, enrich=enrich_with_doi))

    st.markdown("#### Preview (first 20)")
    st.text("\n".join(formatted[:20]))

    if DocxDocument is None:
        st.warning("DOCX export needs python-docx in your environment.")
    else:
        docx_bytes = export_references_to_docx(formatted, title="References")
        st.download_button(
            "Download reformatted References (DOCX)",
            data=docx_bytes,
            file_name=f"references_reformatted_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    st.download_button(
        "Download reformatted References (TXT)",
        data="\n".join(formatted).encode("utf-8"),
        file_name="references_reformatted.txt",
        mime="text/plain"
    )

cite_keys = [c.key for c in cites]
ref_keys = [r.key for r in refs]

missing_keys = sorted(set(cite_keys) - set(ref_keys))
uncited_keys = sorted(set(ref_keys) - set(cite_keys))

cite_counts = Counter(cite_keys)
ref_counts = Counter(ref_keys)

suggestions: Dict[str, List[str]] = {}
if missing_keys and ref_keys:
    for mk in missing_keys:
        matches = process.extract(mk, ref_keys, scorer=fuzz.WRatio, limit=5)
        suggestions[mk] = [f"{m[0]} ({int(m[1])})" for m in matches if m[1] >= 75]

key_to_display_cite = {c.key: c.raw for c in cites}
key_to_full_ref = {r.key: r.raw for r in refs}

df_missing = pd.DataFrame([
    {
        "citation_in_text": key_to_display_cite.get(k, k),
        "count_in_text": cite_counts[k],
        "suggested_matches": ", ".join(suggestions.get(k, []))
    }
    for k in missing_keys
])

df_uncited = pd.DataFrame([
    {
        "reference_full": key_to_full_ref.get(k, k),
        "times_in_references": ref_counts[k],
    }
    for k in uncited_keys
])

df_mismatch = pd.DataFrame()
if style.startswith("APA/Harvard"):
    df_mismatch = find_year_mismatches(cite_keys, ref_keys)

summary = {
    "in_text_citations_found": len(cites),
    "reference_entries_found": len(refs),
    "missing_in_references": len(missing_keys),
    "uncited_references": len(uncited_keys),
}

m1, m2, m3, m4 = st.columns(4)
m1.metric("In-text citations", summary["in_text_citations_found"])
m2.metric("Reference entries", summary["reference_entries_found"])
m3.metric("Missing", summary["missing_in_references"])
m4.metric("Uncited", summary["uncited_references"])

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Cited in-text but missing in References")
    if df_missing.empty:
        st.info("None detected.")
    else:
        st.dataframe(df_missing, use_container_width=True)
    st.download_button(
        "Download missing (CSV)",
        df_missing.to_csv(index=False).encode("utf-8"),
        file_name="missing_in_references.csv",
        mime="text/csv"
    )

with c2:
    st.markdown("### In References but never cited")
    if df_uncited.empty:
        st.info("None detected.")
    else:
        st.dataframe(df_uncited, use_container_width=True)
    st.download_button(
        "Download uncited (CSV)",
        df_uncited.to_csv(index=False).encode("utf-8"),
        file_name="uncited_references.csv",
        mime="text/csv"
    )

st.divider()
st.markdown("### Year mismatches (same author/org, different year)")
if style.startswith("APA/Harvard"):
    if df_mismatch.empty:
        st.success("No year mismatches detected.")
    else:
        st.dataframe(df_mismatch, use_container_width=True)
        st.download_button(
            "Download mismatches (CSV)",
            df_mismatch.to_csv(index=False).encode("utf-8"),
            file_name="year_mismatches.csv",
            mime="text/csv"
        )
else:
    st.info("Year mismatch check applies to author–year styles only.")


# =============================
# Online verification
# =============================
st.divider()
st.subheader("Online verification (Crossref + OpenAlex)")

enable_verify = st.checkbox("Enable online verification", value=False)
use_crossref = st.checkbox("Use Crossref", value=True, disabled=not enable_verify)
use_openalex = st.checkbox("Use OpenAlex", value=True, disabled=not enable_verify)
throttle = st.slider("Throttle seconds between queries", 0.0, 2.0, 0.25, 0.05, disabled=not enable_verify)

df_verify = pd.DataFrame()

if enable_verify:
    test_ok = True
    test_msg = ""
    if use_crossref:
        try:
            _ = _get_json("https://api.crossref.org/works", {"rows": 1})
        except Exception as e:
            test_ok = False
            test_msg = str(e)[:220]

    if (use_crossref and not test_ok) and (not use_openalex):
        st.error("Online verification can’t reach Crossref from this deployment.")
        st.write(f"Error: {test_msg}")
    else:
        rows = []
        progress = st.progress(0.0)  # FIX: always feed 0..1 floats
        total = max(1, len(refs))

        for i, r in enumerate(refs):
            res = verify_reference_online(
                r,
                throttle_s=throttle,
                use_crossref=use_crossref,
                use_openalex=use_openalex
            )
            rows.append({
                "reference": (r.raw[:220] + "…") if len(r.raw) > 220 else r.raw,
                "status": res.get("status", ""),
                "source": res.get("source", ""),
                "score": res.get("score", ""),
                "doi": res.get("doi", ""),
                "matched_year": res.get("matched_year", ""),
                "matched_first_author": res.get("matched_first_author", ""),
                "matched_title": res.get("matched_title", ""),
                "query_used": res.get("query_used", ""),
                "error_crossref": res.get("error_crossref", ""),
                "error_openalex": res.get("error_openalex", ""),
            })
            progress.progress((i + 1) / total)

        df_verify = pd.DataFrame(rows)
        st.dataframe(df_verify, use_container_width=True)
        st.download_button(
            "Download verification (CSV)",
            df_verify.to_csv(index=False).encode("utf-8"),
            file_name="online_verification.csv",
            mime="text/csv"
        )

        flagged = df_verify[df_verify["status"].isin(["not_found", "needs_review"])]
        if len(flagged) > 0:
            st.warning(f"Flagged {len(flagged)} items as Not found or Needs review. Check these first.")


# =============================
# PDF report
# =============================
st.divider()
pdf_bytes = make_pdf_report(style, summary, df_missing, df_uncited, df_mismatch, df_verify)
st.download_button(
    "Download full PDF report",
    data=pdf_bytes,
    file_name="citation_crosscheck_report.pdf",
    mime="application/pdf"
)

# =============================
# Usage statistics display
# =============================
st.divider()
st.subheader("Usage statistics")

stats = load_usage_stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("App runs", stats["app_runs"])
c2.metric("Files processed", stats["files_processed"])
c3.metric("Citations checked", stats["citations_checked"])
c4.metric("References checked", stats["references_checked"])

# =============================
# Debug
# =============================
with st.expander("Extracted items (debug)"):
    tab1, tab2, tab3 = st.tabs(["In-text citations", "Reference entries", "Raw reference splits"])
    with tab1:
        st.dataframe(pd.DataFrame([c.__dict__ for c in cites]), use_container_width=True)
    with tab2:
        st.dataframe(pd.DataFrame([r.__dict__ for r in refs]), use_container_width=True)
    with tab3:
        st.write(f"Split into {len(ref_raw)} raw entries")
        st.text("\n\n---\n\n".join(ref_raw[:20]))
