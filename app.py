# app.py
# Citation Crosschecker (APA/Harvard + IEEE + Vancouver) + Online verification (Crossref + OpenAlex)
# Fixes in this version:
# 1) Keeps ALL occurrences of in-text citations (no forced de-dup), so counts are correct.
# 2) Stronger reference-entry merging for DOCX and PDF (handles wrapped lines better).
# 3) Narrative multi-author citations are overlap-blocked so "Krejcie and Morgan (1970)"
#    will NOT also generate "Morgan (1970)".
# 4) Bare years like "(1991)" are ignored as citations.
# 5) Missing/Uncited tables show FULL raw in-text citations and full reference entries.
# 6) IEEE numeric supported: [1], [1–3], [1,2,5]
# 7) Reconciliation tables: in-text -> reference and reference -> cited-by (APA and numeric)
# 8) Online verification: DOI lookup + bibliographic search against Crossref and OpenAlex

import re
import io
import time
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import streamlit as st
import pandas as pd
import requests
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential


# ----------------------------
# Optional dependencies
# ----------------------------
DOCX_OK = False
try:
    from docx import Document  # python-docx
    DOCX_OK = True
except Exception:
    DOCX_OK = False

PDF_OK = False
try:
    import pdfplumber
    PDF_OK = True
except Exception:
    PDF_OK = False


# ============================
# Constants
# ============================
YEAR = r"(?:1[6-9]\d{2}|20\d{2})(?:[a-z])?"
YEAR_RE = re.compile(rf"\b({YEAR})\b")

REF_HEADINGS = [
    r"^\s*references?\s*(?:list)?\s*$",
    r"^\s*bibliograph(?:y|ies)\s*$",
    r"^\s*works\s+cited\s*$",
    r"^\s*literature\s+cited\s*$",
]

NONCITE_LEADS = {
    "eg", "ie", "see", "cf", "for example", "for instance",
    "chapter", "section", "table", "figure", "eq", "equation", "appendix",
}

BAD_NARRATIVE_PREFIX_WORDS = {
    "traditional", "classical", "analytical", "for", "from", "in", "on", "at", "by",
    "methods", "method", "approach", "approaches", "sample", "size", "power",
    "results", "discussion", "model", "framework",
}

ORG_ALIASES = {
    "who": ["who", "world health organization", "world health organisation"],
    "un": ["un", "united nations", "u.n.", "united nations organisation", "united nations organization"],
    "oecd": ["oecd", "organisation for economic co-operation and development", "organization for economic cooperation and development"],
    "imf": ["imf", "international monetary fund"],
    "world bank": ["world bank", "international bank for reconstruction and development", "ibrd"],
    "unesco": ["unesco", "united nations educational, scientific and cultural organization", "united nations educational scientific and cultural organization"],
    "unicef": ["unicef", "united nations children's fund", "united nations childrens fund"],
}
ORG_ACRONYMS = {k.upper() for k in ["WHO", "UN", "OECD", "IMF", "UNESCO", "UNICEF", "WORLD BANK", "IBRD"]}

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"


# ============================
# Data classes
# ============================
@dataclass
class InTextCitation:
    style: str                  # "author-year" or "numeric"
    raw: str                    # full raw in-text cite
    key: str                    # matching key
    year: Optional[str] = None
    surnames: Optional[Tuple[str, ...]] = None
    number: Optional[int] = None


@dataclass
class ReferenceEntry:
    raw: str
    key: str
    year: Optional[str] = None
    surnames: Optional[Tuple[str, ...]] = None
    number: Optional[int] = None


# ============================
# Normalisation helpers
# ============================
def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00a0", " ").strip())

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_token(s: str) -> str:
    s = (s or "").replace("’", "'")
    s = _strip_accents(s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\-\s'&\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_lead_token(s: str) -> str:
    # for NONCITE_LEADS matching
    s = norm_token(s)
    s = s.replace(".", "")
    return s.strip()

def canon_org(name: str) -> str:
    n = norm_token(name)
    for canon, variants in ORG_ALIASES.items():
        for v in variants:
            if n == norm_token(v):
                return canon
    return n

def is_known_org(text: str) -> bool:
    t = (text or "").strip()
    if t.upper() in ORG_ACRONYMS:
        return True
    c = canon_org(t)
    return c in ORG_ALIASES.keys()

def looks_like_surname(tok: str) -> bool:
    if not tok:
        return False
    t = tok.strip()
    if len(t) < 2:
        return False
    if not re.fullmatch(r"[A-Z][A-Za-z\-']{1,40}", t):
        return False
    if norm_token(t) in BAD_NARRATIVE_PREFIX_WORDS:
        return False
    return True

def clean_surname(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("&", " and ")
    s = re.sub(r"\bet\s+al\.?\b", "", s, flags=re.I)
    s = re.sub(r"[^A-Za-z\-'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    return parts[-1] if parts else ""

def key_author_year(first_surname: str, year: str) -> str:
    return f"au_{norm_token(first_surname)}_{(year or '').lower()}"

def key_numeric(n: int) -> str:
    return f"n_{int(n)}"

def is_bare_year_parenthetical(raw_inside: str) -> bool:
    s = norm_space(raw_inside)
    return bool(re.fullmatch(rf"{YEAR}", s))

def split_semicolons(block: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\s*;\s*", (block or "")) if p.strip()]
    return parts if parts else [block.strip()]


# ============================
# File readers
# ============================
def read_docx_paragraphs(file_bytes: bytes) -> List[str]:
    if not DOCX_OK:
        raise RuntimeError("python-docx not installed")
    doc = Document(io.BytesIO(file_bytes))
    paras = [norm_space(p.text) for p in doc.paragraphs]
    return [p for p in paras if p]

def read_pdf_text(file_bytes: bytes) -> str:
    if not PDF_OK:
        raise RuntimeError("pdfplumber not installed")
    out = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            out.append(p.extract_text() or "")
    return "\n".join(out)


# ============================
# Reference entry merging
# ============================
def _starts_new_reference(line: str) -> bool:
    ln = norm_space(line)
    if not ln:
        return False
    # numeric references
    if re.match(r"^\[\s*\d{1,5}\s*\]", ln):
        return True
    if re.match(r"^\d{1,5}\s*[\.\)]\s+", ln):
        return True
    # author-year ref list often starts with "Surname," or "Organization."
    if re.match(r"^[A-Z][A-Za-z\-']{2,40}\s*,\s*[A-Z]", ln):
        return True
    # org with year early: "World Health Organization. (2020)..."
    if re.match(r"^[A-Z].{2,80}\(\s*" + YEAR + r"\s*\)", ln):
        return True
    return False

def merge_reference_lines(raw_lines: List[str]) -> List[str]:
    lines = [norm_space(x) for x in (raw_lines or []) if norm_space(x)]
    if not lines:
        return []
    merged: List[str] = []
    buf = ""
    for ln in lines:
        if not buf:
            buf = ln
            continue
        if _starts_new_reference(ln):
            merged.append(buf.strip())
            buf = ln
        else:
            buf = (buf + " " + ln).strip()
    if buf:
        merged.append(buf.strip())
    # prune very short noise lines
    merged = [m for m in merged if len(m) >= 10]
    return merged

def split_main_and_references_from_docx(paras: List[str]) -> Tuple[str, List[str], str]:
    heading_idx = None
    heading_line = ""
    for i, line in enumerate(paras):
        s = line.strip()
        for pat in REF_HEADINGS:
            if re.search(pat, s, flags=re.I):
                heading_idx = i
                heading_line = s
                break
        if heading_idx is not None:
            break

    if heading_idx is None:
        return "\n".join(paras), [], "No References heading found in DOCX."

    main = "\n".join(paras[:heading_idx]).strip()
    refs_paras = paras[heading_idx + 1 :]

    ref_lines = merge_reference_lines(refs_paras)
    return main, ref_lines, f"Found heading: {heading_line}"


# ============================
# Reference parsing
# ============================
def parse_reference_author_year(ref_raw: str) -> Optional[ReferenceEntry]:
    r = (ref_raw or "").strip()
    if not r:
        return None

    # remove numeric label if present
    r2 = re.sub(r"^\[\s*\d{1,5}\s*\]\s*", "", r)
    r2 = re.sub(r"^\d{1,5}\s*[\.\)]\s+", "", r2).strip()

    m = re.search(rf"\(\s*({YEAR})\s*\)", r2)
    if not m:
        m2 = re.search(rf"\b({YEAR})\b", r2)
        if not m2:
            return None
        year = m2.group(1)
        pre = r2[: m2.start()].strip()
    else:
        year = m.group(1)
        pre = r2[: m.start()].strip()

    if not pre:
        return None

    # org reference
    if is_known_org(pre) or pre.upper() in ORG_ACRONYMS:
        k = f"org_{canon_org(pre)}_{year.lower()}"
        return ReferenceEntry(raw=r, key=k, year=year, surnames=(pre,), number=None)

    # first author surname (best-effort)
    if "," in pre:
        first = pre.split(",")[0].strip()
    else:
        first = pre.split()[0].strip() if pre.split() else ""

    if not first:
        return None

    k = key_author_year(first, year)
    return ReferenceEntry(raw=r, key=k, year=year, surnames=(first,), number=None)

def parse_reference_numeric(ref_raw: str) -> Optional[ReferenceEntry]:
    r = (ref_raw or "").strip()
    if not r:
        return None

    m = re.match(r"^\s*\[\s*(\d+)\s*\]\s*(.+)$", r)
    if m:
        n = int(m.group(1))
        return ReferenceEntry(raw=r, key=key_numeric(n), number=n)

    m = re.match(r"^\s*(\d+)\s*[\.\)]\s*(.+)$", r)
    if m:
        n = int(m.group(1))
        return ReferenceEntry(raw=r, key=key_numeric(n), number=n)

    m = re.match(r"^\s*(\d+)\s+(.+)$", r)
    if m:
        n = int(m.group(1))
        return ReferenceEntry(raw=r, key=key_numeric(n), number=n)

    return None


# ============================
# In-text extraction (APA/Harvard)
# ============================
def extract_author_year_citations(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []
    txt = text or ""

    taken_spans: List[Tuple[int, int]] = []

    def _overlaps(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
        a, b = span
        for s, e in spans:
            if a < e and b > s:
                return True
        return False

    # --- Parenthetical blocks that contain a year ---
    paren_blocks = re.finditer(rf"\(([^()]*\b{YEAR}\b[^()]*)\)", txt)
    for m in paren_blocks:
        inside = (m.group(1) or "").strip()

        # Reject bare years like "(1991)"
        if is_bare_year_parenthetical(inside):
            continue

        for chunk in split_semicolons(inside):
            c = chunk.strip()
            y_m = YEAR_RE.search(c)
            if not y_m:
                continue
            y = y_m.group(1)

            left = c[: y_m.start()].strip().rstrip(",").strip()
            left_lead = norm_lead_token(left)
            if left_lead in NONCITE_LEADS:
                continue

            # ORG citations
            if is_known_org(left):
                k = f"org_{canon_org(left)}_{y.lower()}"
                out.append(InTextCitation("author-year", f"({norm_space(c)})", k, year=y, surnames=(left,)))
                continue

            # et al.
            if re.search(r"\bet\s+al\.?\b", left, flags=re.I):
                first = clean_surname(left)
                if looks_like_surname(first):
                    k = key_author_year(first, y)
                    out.append(InTextCitation("author-year", f"({norm_space(c)})", k, year=y, surnames=(first,)))
                continue

            # Multi-author inside parentheses: split on commas + and/&
            left2 = left.replace("&", " and ")
            toks = [t.strip() for t in re.split(r"\s+and\s+|,", left2) if t.strip()]
            cand = [t for t in toks if looks_like_surname(t)]
            if not cand:
                continue

            first = cand[0]
            k = key_author_year(first, y)
            out.append(InTextCitation("author-year", f"({norm_space(c)})", k, year=y, surnames=tuple(cand)))

    # --- Narrative multi-author ---
    narr_multi = re.finditer(
        rf"""
        \b
        (?P<authors>
            [A-Z][A-Za-z\-']{{1,40}}
            (?:\s*,\s*[A-Z][A-Za-z\-']{{1,40}})*
            \s*(?:,\s*)?(?:and|&)\s*[A-Z][A-Za-z\-']{{1,40}}
        )
        \s*
        \(\s*(?P<year>{YEAR})\s*\)
        """,
        txt,
        flags=re.VERBOSE,
    )

    for m in narr_multi:
        span = (m.start(), m.end())
        if _overlaps(span, taken_spans):
            continue

        authors_blob = (m.group("authors") or "").strip()
        y = m.group("year")

        first_word = norm_token(authors_blob.split()[0]) if authors_blob.split() else ""
        if first_word in BAD_NARRATIVE_PREFIX_WORDS:
            continue

        blob = authors_blob.replace("&", " and ")
        parts = [p.strip() for p in re.split(r"\s+and\s+|,", blob) if p.strip()]
        cand = [p for p in parts if looks_like_surname(p)]
        if not cand:
            continue

        first = cand[0]
        k = key_author_year(first, y)
        out.append(InTextCitation("author-year", m.group(0), k, year=y, surnames=tuple(cand)))
        taken_spans.append(span)

    # --- Narrative et al. ---
    narr_etal = re.finditer(
        rf"\b(?P<a>[A-Z][A-Za-z\-']{{1,40}})\s+et\s+al\.\s*\(\s*(?P<y>{YEAR})\s*\)",
        txt,
        flags=re.IGNORECASE,
    )
    for m in narr_etal:
        span = (m.start(), m.end())
        if _overlaps(span, taken_spans):
            continue

        first = (m.group("a") or "").strip()
        y = m.group("y")
        if looks_like_surname(first):
            out.append(InTextCitation("author-year", m.group(0), key_author_year(first, y), year=y, surnames=(first,)))
            taken_spans.append(span)

    # --- Narrative single-author ---
    narr_single = re.finditer(
        rf"\b(?P<author>[A-Z][A-Za-z\-']{{1,40}})\s*\(\s*(?P<year>{YEAR})\s*\)",
        txt,
    )
    for m in narr_single:
        span = (m.start(), m.end())
        if _overlaps(span, taken_spans):
            continue

        au = (m.group("author") or "").strip()
        y = m.group("year")

        if norm_token(au) in BAD_NARRATIVE_PREFIX_WORDS:
            continue

        if is_known_org(au):
            k = f"org_{canon_org(au)}_{y.lower()}"
            out.append(InTextCitation("author-year", m.group(0), k, year=y, surnames=(au,)))
        else:
            if looks_like_surname(au):
                out.append(InTextCitation("author-year", m.group(0), key_author_year(au, y), year=y, surnames=(au,)))

    # IMPORTANT: do NOT de-dup here, counts must reflect true frequency.
    return out


# ============================
# Numeric extraction (IEEE + Vancouver)
# ============================
_SUP_DIGITS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}

def _sup_to_int(s: str) -> Optional[int]:
    try:
        digits = "".join(_SUP_DIGITS.get(ch, "") for ch in s)
        return int(digits) if digits else None
    except Exception:
        return None

def _expand_numeric_chunks(inside: str) -> List[int]:
    inside = (inside or "").replace("–", "-")
    chunks = [c.strip() for c in inside.split(",") if c.strip()]
    nums: List[int] = []
    for c in chunks:
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", c)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b and (b - a) <= 2000:
                nums.extend(range(a, b + 1))
        else:
            if c.isdigit():
                nums.append(int(c))
    return nums

def extract_ieee_numeric_citations(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []
    pat = re.compile(r"\[\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*\]")
    for m in pat.finditer(text or ""):
        raw = m.group(0)
        inside = m.group(1)
        for n in _expand_numeric_chunks(inside):
            out.append(InTextCitation("numeric", raw, key_numeric(n), number=n))
    return out

def extract_vancouver_numeric_citations(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []

    paren = re.compile(r"\(\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*\)")
    for m in paren.finditer(text or ""):
        inside = m.group(1)
        # avoid years like (1970)
        if re.fullmatch(rf"{YEAR}", inside.strip()):
            continue
        for n in _expand_numeric_chunks(inside):
            out.append(InTextCitation("numeric", m.group(0), key_numeric(n), number=n))

    out.extend(extract_ieee_numeric_citations(text))

    sup_run = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
    for m in sup_run.finditer(text or ""):
        n = _sup_to_int(m.group(0))
        if n is not None:
            out.append(InTextCitation("numeric", m.group(0), key_numeric(n), number=n))

    return out


# ============================
# Reconciliation
# ============================
# ============================
# Reconciliation (SAFE VERSION)
# ============================
def reconcile_author_year(cites: List[InTextCitation], refs: List[ReferenceEntry]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_key = defaultdict(list)
    for r in refs:
        ref_by_key[r.key].append(r)

    # In-text -> reference
    rows = []
    for c in cites:
        hits = ref_by_key.get(c.key, [])
        if not hits:
            rows.append({"in_text": c.raw, "status": "not_found", "matched_reference": ""})
        elif len(hits) == 1:
            rows.append({"in_text": c.raw, "status": "matched", "matched_reference": hits[0].raw})
        else:
            rows.append({
                "in_text": c.raw,
                "status": f"ambiguous ({len(hits)})",
                "matched_reference": " || ".join(h.raw[:200] for h in hits),
            })

    df_c2r = pd.DataFrame(rows, columns=["in_text", "status", "matched_reference"])

    # Reference -> cited by
    cite_group = defaultdict(list)
    for c in cites:
        cite_group[c.key].append(c.raw)

    ref_rows = []
    for r in refs:
        cited_by = cite_group.get(r.key, [])
        ref_rows.append({
            "reference": r.raw,
            "times_cited": int(len(cited_by)),
            "cited_by": "; ".join(cited_by[:12]) + (" ..." if len(cited_by) > 12 else ""),
        })

    df_r2c = pd.DataFrame(ref_rows, columns=["reference", "times_cited", "cited_by"])
    if not df_r2c.empty:
        df_r2c = df_r2c.sort_values(["times_cited"], ascending=False)

    return df_c2r, df_r2c


def reconcile_numeric(cites: List[InTextCitation], refs: List[ReferenceEntry]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_key = {r.key: r for r in refs}

    # In-text -> reference
    rows = []
    for c in cites:
        r = ref_by_key.get(c.key)
        if not r:
            rows.append({"in_text": c.raw, "status": "not_found", "matched_reference": ""})
        else:
            rows.append({"in_text": c.raw, "status": "matched", "matched_reference": r.raw})

    df_c2r = pd.DataFrame(rows, columns=["in_text", "status", "matched_reference"])

    # Reference -> cited by
    cite_group = defaultdict(list)
    for c in cites:
        cite_group[c.key].append(c.raw)

    ref_rows = []
    for r in refs:
        cited_by = cite_group.get(r.key, [])
        ref_rows.append({
            "reference": r.raw,
            "times_cited": int(len(cited_by)),
            "cited_by": "; ".join(cited_by[:18]) + (" ..." if len(cited_by) > 18 else ""),
        })

    df_r2c = pd.DataFrame(ref_rows, columns=["reference", "times_cited", "cited_by"])
    if not df_r2c.empty:
        df_r2c = df_r2c.sort_values(["times_cited"], ascending=False)

    return df_c2r, df_r2c



   

# ============================
# Missing + Uncited (full raw, correct counts)
# ============================
def build_missing_uncited(cites: List[InTextCitation], refs: List[ReferenceEntry]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cite_count_by_raw = Counter([c.raw for c in cites])
    cite_key_by_raw: Dict[str, str] = {}
    for c in cites:
        # first observed key for this raw
        cite_key_by_raw.setdefault(c.raw, c.key)

    ref_key_set = set([r.key for r in refs])

    missing_rows = []
    for raw, cnt in cite_count_by_raw.items():
        k = cite_key_by_raw.get(raw, "")
        if k and (k not in ref_key_set):
            missing_rows.append({"citation_in_text": raw, "count_in_text": int(cnt)})

    df_missing = (
        pd.DataFrame(missing_rows).sort_values(["count_in_text", "citation_in_text"], ascending=[False, True]).reset_index(drop=True)
        if missing_rows
        else pd.DataFrame(columns=["citation_in_text", "count_in_text"])
    )

    cite_key_set = set([c.key for c in cites])
    uncited_rows = [{"reference_full": r.raw} for r in refs if r.key not in cite_key_set]
    df_uncited = pd.DataFrame(uncited_rows) if uncited_rows else pd.DataFrame(columns=["reference_full"])

    summary = {
        "in_text_citations_found": int(len(cites)),
        "unique_in_text_citations": int(len(cite_count_by_raw)),
        "reference_entries_found": int(len(refs)),
        "missing_in_references": int(len(df_missing)),
        "uncited_references": int(len(df_uncited)),
    }
    return df_missing, df_uncited, summary


# ============================
# Online verification helpers
# ============================
def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "citation-crosscheck/2.0 (streamlit)"})
    return s

SESSION = build_session()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _get_json(url: str, params: dict) -> dict:
    r = SESSION.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def extract_doi(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"https?://doi\.org/(10\.\d{4,9}/[^\s<>\"]+)", text, flags=re.I)
    if m:
        doi = m.group(1)
    else:
        m2 = re.search(r"(10\.\d{4,9}/[^\s<>\"]+)", text, flags=re.I)
        if not m2:
            return None
        doi = m2.group(1)

    doi = doi.strip().strip(").,;:]}>\"'")
    doi = re.sub(r"&quot;|&gt;|&lt;|&amp;", "", doi)
    doi = doi.strip().strip(").,;:]}>\"'")
    return doi if doi.lower().startswith("10.") else None

def _as_int_year(y: Optional[str]) -> Optional[int]:
    if not y:
        return None
    m = re.search(r"(16|17|18|19|20)\d{2}", str(y))
    return int(m.group(0)) if m else None

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
    for key in ["issued", "published-print", "published-online", "created"]:
        dp = (it.get(key, {}) or {}).get("date-parts", [])
        if dp and dp[0]:
            try:
                return int(dp[0][0])
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
    return abs(ref_year - cand_year) == 1

def guess_title_snippet(ref: str) -> str:
    r = norm_space(ref)
    r = re.sub(r"https?://doi\.org/\S+", " ", r, flags=re.I)
    r = re.sub(r"\b10\.\d{4,9}/\S+", " ", r, flags=re.I)
    r = re.sub(rf"\(.*?\b{YEAR}\b.*?\)", " ", r)
    r = norm_space(r)
    r = re.sub(r"^[^\.]{1,220}\.\s*", " ", r)
    r = norm_space(r)
    words = r.split()
    return " ".join(words[:22])[:260]

def verify_reference_online(
    ref_obj: ReferenceEntry,
    throttle_s: float = 0.25,
    use_crossref: bool = True,
    use_openalex: bool = True,
) -> dict:
    ref_entry = ref_obj.raw or ""
    doi = extract_doi(ref_entry)

    ref_year = _as_int_year(ref_obj.year)
    ref_surname = ""
    if ref_obj.key.startswith("au_") and ref_obj.surnames:
        ref_surname = ref_obj.surnames[0] or ""
    who_is_org = ref_obj.key.startswith("org_")

    title_snip = guess_title_snippet(ref_entry)
    time.sleep(max(0.0, float(throttle_s or 0.0)))

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
    if ref_surname:
        parts.append(ref_surname)
    if ref_year:
        parts.append(str(ref_year))
    if title_snip:
        parts.append(title_snip)
    query = " ".join(parts).strip() or ref_entry[:200]

    best = {
        "status": "not_found",
        "source": "",
        "score": 0,
        "doi": doi or "",
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
                    if ref_surname and cand_fam and not author_match_ok(ref_surname, cand_fam):
                        continue
                    if ref_year and cand_year and not year_match_ok(ref_year, cand_year):
                        continue
                else:
                    if ref_year and cand_year and not year_match_ok(ref_year, cand_year):
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
                    if ref_surname and cand_fam and not author_match_ok(ref_surname, cand_fam):
                        continue
                    if ref_year and cand_year and not year_match_ok(ref_year, cand_year):
                        continue
                else:
                    if ref_year and cand_year and not year_match_ok(ref_year, cand_year):
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


# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Citation Crosschecker", layout="wide")
st.title("Citation Crosschecker")

st.markdown(
    """
<div style="margin-top:6px; margin-bottom:10px; padding:10px 12px; border:1px solid #e6e6e6; border-radius:10px;">
  <div style="font-size:13px; font-weight:600;">© Prof. Anokye M. Adam</div>
  <div style="font-size:12px; margin-top:6px; line-height:1.35;">
    Disclaimer: This tool can make mistakes. Verify results against your manuscript and style guide before submission.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Upload DOCX or PDF", type=["docx", "pdf"])
with col2:
    style = st.selectbox(
        "Citation style",
        ["APA/Harvard (author–year)", "IEEE (numeric [1])", "Vancouver (numeric (1)/superscript)"],
        index=0,
    )

if uploaded is None:
    st.info("Upload a DOCX or PDF to begin.")
    st.stop()

file_bytes = uploaded.read()
name = uploaded.name.lower()

main_text = ""
ref_lines: List[str] = []
ref_msg = ""

try:
    if name.endswith(".docx"):
        paras = read_docx_paragraphs(file_bytes)
        main_text, ref_lines, ref_msg = split_main_and_references_from_docx(paras)
    else:
        full = read_pdf_text(file_bytes)
        lines = [ln for ln in full.splitlines() if ln is not None]
        idx = None
        for i, ln in enumerate(lines):
            s = (ln or "").strip()
            if any(re.search(p, s, flags=re.I) for p in REF_HEADINGS):
                idx = i
                ref_msg = f"Found heading: {s}"
                break
        if idx is None:
            main_text = full
            ref_lines = []
            ref_msg = "No References heading found in PDF. (PDF extraction is less reliable.)"
        else:
            main_text = "\n".join(lines[:idx]).strip()
            ref_block = "\n".join(lines[idx + 1 :]).strip()
            raw_lines = [ln.strip() for ln in ref_block.splitlines() if ln.strip()]
            ref_lines = merge_reference_lines(raw_lines)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("References detection")
st.write(ref_msg)

if not ref_lines:
    st.warning("References section looks empty or not detected. Add a 'References' heading in the DOCX for best results.")
    with st.expander("Paste References manually"):
        pasted_refs = st.text_area("Paste your References section here", height=220)
        if pasted_refs.strip():
            raw_lines = [x.strip() for x in pasted_refs.splitlines() if x.strip()]
            ref_lines = merge_reference_lines(raw_lines)

st.caption(f"Main text length: {len(main_text):,} chars | Reference entries detected: {len(ref_lines):,}")

with st.expander("Preview detected References (first 50)"):
    st.write("\n\n".join(ref_lines[:50]) if ref_lines else "No references detected.")

# Parse cites + refs based on style
if style.startswith("APA/Harvard"):
    cites = extract_author_year_citations(main_text)
    refs = [parse_reference_author_year(r) for r in ref_lines]
    refs = [r for r in refs if r is not None]
elif style.startswith("IEEE"):
    cites = extract_ieee_numeric_citations(main_text)
    refs = [parse_reference_numeric(r) for r in ref_lines]
    refs = [r for r in refs if r is not None]
else:
    cites = extract_vancouver_numeric_citations(main_text)
    refs = [parse_reference_numeric(r) for r in ref_lines]
    refs = [r for r in refs if r is not None]

df_missing, df_uncited, summary = build_missing_uncited(cites, refs)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("In-text citations", summary["in_text_citations_found"])
m2.metric("Unique in-text", summary["unique_in_text_citations"])
m3.metric("Reference entries", summary["reference_entries_found"])
m4.metric("Missing", summary["missing_in_references"])
m5.metric("Uncited", summary["uncited_references"])

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Cited in-text but missing in References")
    if df_missing.empty:
        st.success("None detected.")
    else:
        st.dataframe(df_missing, width="stretch", height=420)
        st.download_button(
            "Download missing (CSV)",
            df_missing.to_csv(index=False).encode("utf-8"),
            file_name="missing_in_references.csv",
            mime="text/csv",
        )

with c2:
    st.markdown("### In References but never cited (full list)")
    if df_uncited.empty:
        st.success("None detected.")
    else:
        st.dataframe(df_uncited, width="stretch", height=420)
        st.download_button(
            "Download uncited (CSV)",
            df_uncited.to_csv(index=False).encode("utf-8"),
            file_name="uncited_references.csv",
            mime="text/csv",
        )

st.divider()
st.subheader("Reconciliation (maps in-text citations to exact references)")

if style.startswith("APA/Harvard"):
    df_c2r, df_r2c = reconcile_author_year(cites, refs)
else:
    df_c2r, df_r2c = reconcile_numeric(cites, refs)

left, right = st.columns(2)
with left:
    st.markdown("### In-text → Reference")
    st.dataframe(df_c2r, width="stretch", height=420)
    st.download_button(
        "Download in-text to reference mapping (CSV)",
        df_c2r.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_intext_to_reference.csv",
        mime="text/csv",
    )

with right:
    st.markdown("### Reference → Cited by")
    st.dataframe(df_r2c, width="stretch", height=420)
    st.download_button(
        "Download reference to in-text mapping (CSV)",
        df_r2c.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_reference_to_intext.csv",
        mime="text/csv",
    )

# ============================
# Online verification (Crossref + OpenAlex)
# ============================
st.divider()
st.subheader("Online verification (Crossref + OpenAlex)")

enable_verify = st.checkbox("Enable online verification", value=False)
use_crossref = st.checkbox("Use Crossref", value=True, disabled=not enable_verify)
use_openalex = st.checkbox("Use OpenAlex", value=True, disabled=not enable_verify)
throttle = st.slider("Throttle seconds between queries", 0.0, 2.0, 0.25, 0.05, disabled=not enable_verify)
max_to_check = st.number_input(
    "Max references to verify (for speed)",
    min_value=10,
    max_value=5000,
    value=min(200, max(10, len(refs))),
    step=10,
    disabled=not enable_verify
)

df_verify = pd.DataFrame()

if enable_verify:
    test_ok = True
    test_msg = ""
    if use_crossref:
        try:
            _ = _get_json(CROSSREF_API, {"rows": 1})
        except Exception as e:
            test_ok = False
            test_msg = str(e)[:220]

    if (use_crossref and not test_ok) and (not use_openalex):
        st.error("Online verification can’t reach Crossref from this deployment.")
        st.write(f"Error: {test_msg}")
    else:
        rows = []
        progress = st.progress(0)
        work = refs[: int(max_to_check)]
        total = len(work) if work else 1

        for i, r in enumerate(work):
            res = verify_reference_online(
                r,
                throttle_s=float(throttle),
                use_crossref=bool(use_crossref),
                use_openalex=bool(use_openalex),
            )
            rows.append(
                {
                    "reference": (r.raw[:240] + "…") if len(r.raw) > 240 else r.raw,
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
                }
            )
            progress.progress(int((i + 1) / total * 100))

        df_verify = pd.DataFrame(rows)
        st.dataframe(df_verify, width="stretch", height=460)
        st.download_button(
            "Download verification (CSV)",
            df_verify.to_csv(index=False).encode("utf-8"),
            file_name="online_verification.csv",
            mime="text/csv",
        )

        flagged = df_verify[df_verify["status"].isin(["not_found", "needs_review"])]
        if len(flagged) > 0:
            st.warning(f"Flagged {len(flagged)} items as Not found or Needs review. Check these first.")
            with st.expander("Show flagged only"):
                st.dataframe(flagged, width="stretch", height=360)

with st.expander("Diagnostics"):
    st.markdown("#### Sample extracted in-text citations (first 120)")
    st.dataframe(pd.DataFrame([c.__dict__ for c in cites[:120]]), width="stretch")
    st.markdown("#### Sample parsed references (first 120)")
    st.dataframe(pd.DataFrame([r.__dict__ for r in refs[:120]]), width="stretch")
    if enable_verify and not df_verify.empty:
        st.markdown("#### Sample verification output (first 60)")
        st.dataframe(df_verify.head(60), width="stretch")



