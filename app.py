import re
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Iterable
from collections import Counter, defaultdict
from io import BytesIO

import streamlit as st
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from tenacity import retry, stop_after_attempt, wait_exponential

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# Optional readers
try:
    import docx
except Exception:
    docx = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# =============================
# Constants
# =============================
YEAR = r"(?:19|20)\d{2}[a-z]?"  # allows 2020a
CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"

# Heading detection that matches: Reference |, REFERENCES:, Reference List, etc.
REF_HEADINGS = [
    r"^\s*references?\s*(?:list)?\s*[:|]?\s*$",
    r"^\s*bibliograph(?:y|ies)\s*[:|]?\s*$",
    r"^\s*works\s+cited\s*[:|]?\s*$",
    r"^\s*literature\s+cited\s*[:|]?\s*$",
]

# Known organisation aliases (expand as needed)
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


# =============================
# Data classes
# =============================
@dataclass
class InTextCitation:
    style: str                  # author-year | org-year | numeric
    raw: str
    key: str                    # internal key (stable)
    pretty: str                 # display e.g., "Kofi, 2000"
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
    """
    blob examples:
      "Mensah" / "Mensah & Boateng" / "Mensah and Boateng" / "Mensah et al."
      "Al-Majali, Alsarayreh & Alqaralleh"
    Returns first surname token.
    """
    if not blob:
        return None
    b = blob.strip()
    b = re.sub(r"^(see|e\.g\.|cf\.)\s+", "", b, flags=re.IGNORECASE).strip()

    # remove trailing et al.
    b = re.sub(r"\s+et\s+al\.?$", "", b, flags=re.IGNORECASE).strip()

    # prefer first token before comma
    first = b.split(",")[0].strip()

    # then split on and/&
    first = re.split(r"\s+(?:and|&)\s+", first, flags=re.IGNORECASE)[0].strip()

    m = re.search(r"([A-Z][A-Za-z\-']+)", first)
    return m.group(1) if m else None
NONNAME_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "as", "by",
    "from", "into", "at", "than", "that", "this", "these", "those",
    "recent", "minimum", "maximum", "mathematical", "basis", "adopting", "adopt", "model",
    "framework", "table", "figure", "chapter", "section", "appendix", "equation",
    "definition", "method", "methods", "result", "results", "discussion", "similarly", "moreover", "however", "therefore", "thus", "also", "further", "additionally",
"consequently", "notably", "specifically", "generally", "overall", "first", "second", "finally"
}

def looks_like_person_surname(token: str) -> bool:
    if not token:
        return False
    t = token.strip()

    # reject possessive like Adam's / Adam’s
    if re.search(r"(?:'s|’s)$", t, flags=re.IGNORECASE):
        return False

    # basic surname pattern (allows Al-Majali)
    if not re.fullmatch(r"[A-Z][A-Za-z\-']{1,40}", t):
        return False

    # reject common non-name words
    if norm_token(t) in NONNAME_STOPWORDS:
        return False

    return True


# =============================
# DOCX block iterator (keeps order of paragraphs and tables)
# =============================
def iter_block_items(document) -> Iterable[Tuple[str, str]]:
    """
    Yields ("p", paragraph_text) or ("t", table_row_text) in document order.
    This avoids the common bug where references are in tables and ordering breaks.
    """
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
    """
    Returns (best_line_index, confidence_score 0..1).
    Confidence is based on how strong the window scoring peak is.
    """
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
# In-text citation extraction (APA/Harvard)
# =============================
def extract_author_year_citations(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []

    # -----------------------------
    # Narrative citations (NON-parenthetical) for PEOPLE only:
    #   Kofi (2023)
    #   Kofi and Ama (2023)
    #   Kofi, Ama and Kwame (2023)
    #   Kofi et al. (2023)
    #
    # IMPORTANT: We intentionally DO NOT accept "org-ish" catch-all here,
    # because it creates false positives like "Mathematical (2020)".
    # Organisations are handled separately (only known orgs).
    # -----------------------------
    narr_people = re.compile(
        rf"""
        \b
        (?P<authors>
            [A-Z][A-Za-z\-']+                                  # first surname
            (?:\s*,\s*[A-Z][A-Za-z\-']+)*                      # optional ", Ama, Kwame"
            (?:\s*,?\s*(?:and|&)\s*[A-Z][A-Za-z\-']+)?         # optional "and Yaw"
            |
            [A-Z][A-Za-z\-']+\s+et\s+al\.                      # "Kofi et al."
        )
        \s*
        \(\s*(?P<year>{YEAR})\s*\)
        """,
        flags=re.VERBOSE
    )

    for m in narr_people.finditer(text):
        authors_blob = m.group("authors").strip()
        y = m.group("year")

        # strip trailing et al. for keying
        authors_blob_clean = re.sub(r"\s+et\s+al\.\s*$", "", authors_blob, flags=re.IGNORECASE).strip()
        first = _first_surname_from_author_blob(authors_blob_clean) or authors_blob_clean.split(",")[0].strip()

        # Block false positives like "The (2020)", "Mathematical (2020)", "Minimum (1991)", "Adam's (2020)"
        if not looks_like_person_surname(first):
            continue

        key = key_author_year(first, y)
        pretty = f"{titleish(first)}, {y}"
        out.append(InTextCitation("author-year", m.group(0), key, pretty, first, y))

    # -----------------------------
    # Narrative citations for ORGANISATIONS (STRICT: known orgs only):
    #   UNCTAD (2025)
    #   World Bank (2023)
    # -----------------------------
    narr_org = re.compile(rf"\b([A-Z][A-Za-z&.\- ]{{2,}}?)\s*\(\s*({YEAR})\s*\)")
    for m in narr_org.finditer(text):
        org, y = m.group(1).strip(), m.group(2)

        # Stop false positives like "Saslavsky Shepherd (2014)"
        if looks_like_two_authors(org):
            continue

        # Only accept organisations we recognise
        if is_known_org(org):
            key = key_org_year(org, y)
            pretty = f"{titleish(org)}, {y}"
            out.append(InTextCitation("org-year", m.group(0), key, pretty, org, y))

    # -----------------------------
    # Parenthetical blocks:
    #   (Kofi, 2000; Mensah & Boateng, 2001; UNCTAD, 2025)
    # -----------------------------
    paren_block = re.compile(rf"\(([^()]*?\b{YEAR}\b[^()]*)\)")
    for m in paren_block.finditer(text):
        block = m.group(1)
        parts = [p.strip() for p in block.split(";") if p.strip()]

        for p in parts:
            yrm = re.search(rf"\b({YEAR})\b", p)
            if not yrm:
                continue
            y = yrm.group(1)

            # left side in "Name, 2020"
            left = p
            m_left = re.search(rf"^(.+?)\s*,\s*{YEAR}\b", p)
            if m_left:
                left = m_left.group(1).strip()

            # Organisation inside parentheses: (UNCTAD, 2025)
            if is_known_org(left) and not looks_like_two_authors(left):
                key = key_org_year(left, y)
                pretty = f"{titleish(left)}, {y}"
                out.append(InTextCitation("org-year", f"({p})", key, pretty, left, y))
                continue

            # People: extract first surname and validate
            first_author = _first_surname_from_author_blob(left) or _first_surname_from_author_blob(p)
            if not first_author:
                continue

            # if the extracted token isn't a plausible surname, skip it
            if not looks_like_person_surname(first_author):
                continue

            key = key_author_year(first_author, y)
            pretty = f"{titleish(first_author)}, {y}"
            out.append(InTextCitation("author-year", f"({p})", key, pretty, first_author, y))

    return out


# =============================
# Numeric citation extraction
# =============================
def extract_ieee_numeric_citations(text: str) -> List[InTextCitation]:
    out = []
    pat = re.compile(r"\[(\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*\]")
    for m in pat.finditer(text):
        raw = m.group(0)
        inside = m.group(1)
        chunks = [c.strip() for c in inside.split(",")]
        nums = []
        for c in chunks:
            r = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", c)
            if r:
                a, b = int(r.group(1)), int(r.group(2))
                if a <= b and (b - a) <= 2000:
                    nums.extend(range(a, b + 1))
            else:
                if c.isdigit():
                    nums.append(int(c))
        for n in nums:
            out.append(InTextCitation("numeric", raw, key_numeric(n), f"[{n}]", number=n))
    return out

def extract_vancouver_parentheses_numeric(text: str) -> List[InTextCitation]:
    out = []
    pat = re.compile(r"\((\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*\)")
    for m in pat.finditer(text):
        raw = m.group(0)
        inside = m.group(1)

        if re.fullmatch(rf"\s*{YEAR}\s*", inside):
            continue

        chunks = [c.strip() for c in inside.split(",")]
        nums = []
        for c in chunks:
            r = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", c)
            if r:
                a, b = int(r.group(1)), int(r.group(2))
                if a <= b and (b - a) <= 2000:
                    nums.extend(range(a, b + 1))
            else:
                if c.isdigit():
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
    entries = []
    buf = ""

    start_pat = re.compile(
        r"^\s*(\[\d+\]|\d+[\.\)])\s+"                  # numbered start
        r"|^\s*[A-Z][A-Za-z\-']+\s*,"                  # surname,
        r"|^\s*[A-Z][A-Z&\- ]{2,}\."                   # ORG.
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

    # Organisation: UNCTAD. (2025). ...
    org_m = re.search(rf"^([A-Z][A-Z&\- ]{{2,}})\.\s*\(\s*({YEAR})\s*\)", e)
    if org_m:
        org, y = org_m.group(1).strip(), org_m.group(2)
        if is_known_org(org):
            key = key_org_year(org, y)
            pretty = f"{titleish(org)}, {y}"
            return ReferenceEntry(entry, key, pretty, org, y)

    # Organisation: World Bank (2023). ... OR World Bank. (2023). ...
    org2 = re.search(rf"^([A-Z][A-Za-z&.\- ]{{2,}}?)\.?\s*\(\s*({YEAR})\s*\)", e)
    if org2:
        org, y = org2.group(1).strip(), org2.group(2)
        if is_known_org(org) and not looks_like_two_authors(org):
            key = key_org_year(org, y)
            pretty = f"{titleish(org)}, {y}"
            return ReferenceEntry(entry, key, pretty, org, y)

    # Author-year: Surname, X. (2020). ...
    m = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\(\s*({YEAR})\s*\)", e)
    if m:
        au, y = m.group(1), m.group(2)
        key = key_author_year(au, y)
        pretty = f"{titleish(au)}, {y}"
        return ReferenceEntry(entry, key, pretty, au, y)

    # Harvard-ish: Surname, X. ... 2020 ...
    m2 = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\b({YEAR})\b", e)
    if m2:
        au, y = m2.group(1), m2.group(2)
        key = key_author_year(au, y)
        pretty = f"{titleish(au)}, {y}"
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
# Online verification
# =============================
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
    """
    Safer DOI extraction:
    - supports https://doi.org/...
    - strips trailing punctuation like ")."
    """
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


# ---- Strict metadata helpers ----
def _as_int_year(y: Optional[str]) -> Optional[int]:
    if not y:
        return None
    m = re.search(r"(19|20)\d{2}", str(y))
    return int(m.group(0)) if m else None

def crossref_item_year(it: dict) -> Optional[int]:
    issued = it.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        try:
            return int(issued[0][0])
        except Exception:
            pass
    created = it.get("created", {}).get("date-parts", [])
    if created and created[0]:
        try:
            return int(created[0][0])
        except Exception:
            pass
    published_print = it.get("published-print", {}).get("date-parts", [])
    if published_print and published_print[0]:
        try:
            return int(published_print[0][0])
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

def author_match_ok(ref_surname: str, cand_surname: str) -> bool:
    if not ref_surname or not cand_surname:
        return False
    return norm_token(ref_surname) == norm_token(cand_surname)

def year_match_ok(ref_year: Optional[int], cand_year: Optional[int]) -> bool:
    if ref_year is None or cand_year is None:
        return False
    if ref_year == cand_year:
        return True
    # allow ±1 for online-first vs print, editions
    return abs(ref_year - cand_year) == 1


def guess_title_snippet(ref: str) -> str:
    r = _strip_leading_numbering(ref)
    # drop year-parentheses chunk
    t = re.sub(rf"\(.*?{YEAR}.*?\)", " ", r)
    # drop first segment up to first period (usually author list)
    t = re.sub(r"^[^\.]{1,180}\.\s*", " ", t)
    t = norm_spaces(t)
    words = t.split()
    return " ".join(words[:18])[:220]  # slightly longer improves title scoring


def verify_reference_online(
    ref_obj: ReferenceEntry,
    throttle_s: float = 0.2,
    use_crossref: bool = True,
    use_openalex: bool = True
) -> dict:
    """
    Strict verification:
    - If DOI exists: verify by DOI only (no search guessing)
    - Else: require (author/org + year) agreement before accepting a candidate
    - Score mainly on title similarity after passing the gates
    """
    ref_entry = ref_obj.raw or ""
    doi = extract_doi(ref_entry)

    ref_year = _as_int_year(ref_obj.year)
    who = (ref_obj.author_or_org or "").strip()
    who_is_org = ref_obj.key.startswith("org_")

    # Enforce only plausible person surname for author-year items
    ref_surname = who if (who and not who_is_org and looks_like_person_surname(who)) else ""

    title_snip = guess_title_snippet(ref_entry)

    # ---- DOI first (strongest) ----
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
        }

    # ---- Build query (high signal, less noise) ----
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

    # ---- Crossref (strict gates) ----
    if use_crossref:
        try:
            items = crossref_search(query, rows=10)
            for it in items:
                cand_year = crossref_item_year(it)
                cand_fam = crossref_first_author_family(it)
                cand_title = crossref_title(it)
                cand_doi = (it.get("DOI") or "").strip()

                # Gate 1: author/org agreement
                if who_is_org:
                    # for org refs, we can't reliably match author family, so skip author gate
                    pass
                else:
                    if ref_surname and not author_match_ok(ref_surname, cand_fam):
                        continue

                # Gate 2: year agreement (if we have a parsed year)
                if ref_year and not year_match_ok(ref_year, cand_year):
                    continue

                # Score: title similarity dominates
                score = fuzz.WRatio(title_snip, cand_title) if title_snip and cand_title else 80
                if score > best["score"]:
                    best = {
                        "status": "verified" if score >= 82 else "needs_review",
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

    # ---- OpenAlex (strict gates) ----
    if use_openalex:
        try:
            items = openalex_search(query, per_page=10)
            for it in items:
                cand_year = openalex_year(it)
                cand_fam = openalex_first_author_family(it)
                cand_title = openalex_title(it)
                cand_doi = openalex_doi(it)

                if who_is_org:
                    pass
                else:
                    if ref_surname and not author_match_ok(ref_surname, cand_fam):
                        continue

                if ref_year and not year_match_ok(ref_year, cand_year):
                    continue

                score = fuzz.WRatio(title_snip, cand_title) if title_snip and cand_title else 80
                if score > best["score"]:
                    best = {
                        "status": "verified" if score >= 82 else "needs_review",
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
with st.expander("Sanity check: end of main text"):
    st.text(main_text[-3500:] if main_text else "")

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

cite_keys = [c.key for c in cites]
ref_keys = [r.key for r in refs]

missing_keys = sorted(set(cite_keys) - set(ref_keys))
uncited_keys = sorted(set(ref_keys) - set(cite_keys))

cite_counts = Counter(cite_keys)
ref_counts = Counter(ref_keys)

# Suggestions for missing keys (fuzzy on keys)
suggestions: Dict[str, List[str]] = {}
if missing_keys and ref_keys:
    for mk in missing_keys:
        matches = process.extract(mk, ref_keys, scorer=fuzz.WRatio, limit=5)
        suggestions[mk] = [f"{m[0]} ({int(m[1])})" for m in matches if m[1] >= 75]

# User-friendly labels
key_to_pretty_cite = {c.key: c.pretty for c in cites}
key_to_pretty_ref = {r.key: r.pretty for r in refs}

df_missing = pd.DataFrame([
    {
        "citation": key_to_pretty_cite.get(k, k),
        "count_in_text": cite_counts[k],
        "internal_key": k,
        "suggested_matches": ", ".join(suggestions.get(k, []))
    }
    for k in missing_keys
])

df_uncited = pd.DataFrame([
    {
        "reference": key_to_pretty_ref.get(k, k),
        "times_in_references": ref_counts[k],
        "internal_key": k
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
        st.info("On Streamlit Cloud, redeploy and avoid rapid repeated queries. Also check if your network blocks outbound requests.")
    else:
        rows = []
        progress = st.progress(0)
        total = len(refs) if refs else 1

        for i, r in enumerate(refs):
            res = verify_reference_online(
                r,
                throttle_s=throttle,
                use_crossref=use_crossref,
                use_openalex=use_openalex
            )
            rows.append({
                "reference": r.raw[:220],
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
            progress.progress(int((i + 1) / total * 100))

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



