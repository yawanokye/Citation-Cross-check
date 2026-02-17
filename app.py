# app.py
# Citation Crosschecker (APA/Harvard + IEEE + Vancouver)
# Fixes included:
# - Narrative citations now keep full strings like "Krejcie and Morgan (1970)"
#   and "Bartlett, Kotrlik, and Higgins (2001)" (not just Morgan/Higgins).
# - Stops counting bare years like "(1991)" as citations.
# - Stops counting random sentences as citations.
# - Shows uncited reference COUNT and full list.
# - Adds IEEE style support ([1], [1-3], [1,2,5]).
# - Adds reconciliation tables: in-text -> reference, reference -> cited by.

import re
import io
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

import streamlit as st
import pandas as pd

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
    "e.g", "i.e", "see", "cf", "for example", "for instance",
    "chapter", "section", "table", "figure", "eq", "equation", "appendix",
}

# If narrative capture grabs long phrases, reject unless it looks like names/org
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
    return re.sub(r"\s+", " ", (s or "").strip())

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
    parts = [p.strip() for p in (block or "").split(";") if p.strip()]
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

    refs: List[str] = []
    buf = ""
    for p in refs_paras:
        if not buf:
            buf = p
            continue
        if re.match(r"^\s*([a-z]|\d|\()", p):
            buf = (buf + " " + p).strip()
        else:
            refs.append(buf.strip())
            buf = p
    if buf:
        refs.append(buf.strip())

    refs = [r for r in refs if len(r) >= 10]
    return main, refs, f"Found heading: {heading_line}"


# ============================
# Reference parsing
# ============================
def parse_reference_author_year(ref_raw: str) -> Optional[ReferenceEntry]:
    r = (ref_raw or "").strip()
    if not r:
        return None

    m = re.search(rf"\(\s*({YEAR})\s*\)", r)
    if not m:
        m2 = re.search(rf"\b({YEAR})\b", r)
        if not m2:
            return None
        year = m2.group(1)
        pre = r[: m2.start()].strip()
    else:
        year = m.group(1)
        pre = r[: m.start()].strip()

    if is_known_org(pre) or pre.upper() in ORG_ACRONYMS:
        k = f"org_{canon_org(pre)}_{year.lower()}"
        return ReferenceEntry(raw=r, key=k, year=year, surnames=(pre,), number=None)

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

    # --- Parenthetical blocks that contain a year ---
    paren_blocks = re.finditer(rf"\(([^()]*\b{YEAR}\b[^()]*)\)", txt)
    for m in paren_blocks:
        inside = m.group(1).strip()

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
            left_norm = norm_token(left)
            if left_norm in NONCITE_LEADS:
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

    # ============================
    # Narrative citations (FIXED)
    # ============================

    # 1) Multi-author narrative:
    #    Krejcie and Morgan (1970)
    #    Bartlett, Kotrlik, and Higgins (2001)
    #    Bartlett, Kotrlik, & Higgins (2001)
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
        authors_blob = m.group("authors").strip()
        y = m.group("year")

        # Reject if it starts with a narrative word (rare but protects false positives)
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

        # Keep FULL raw, not last surname
        out.append(InTextCitation("author-year", m.group(0), k, year=y, surnames=tuple(cand)))

    # 2) "et al." narrative: A et al. (YEAR)
    narr_etal = re.finditer(
        rf"\b(?P<a>[A-Z][A-Za-z\-']{{1,40}})\s+et\s+al\.\s*\(\s*(?P<y>{YEAR})\s*\)",
        txt,
        flags=re.IGNORECASE,
    )
    for m in narr_etal:
        first = m.group("a").strip()
        y = m.group("y")
        if looks_like_surname(first):
            out.append(InTextCitation("author-year", m.group(0), key_author_year(first, y), year=y, surnames=(first,)))

    # 3) Single-author narrative: A (YEAR) or ORG (YEAR)
    narr_single = re.finditer(
        rf"\b(?P<author>[A-Z][A-Za-z\-']{{1,40}})\s*\(\s*(?P<year>{YEAR})\s*\)",
        txt,
    )
    for m in narr_single:
        au = m.group("author").strip()
        y = m.group("year")

        if norm_token(au) in BAD_NARRATIVE_PREFIX_WORDS:
            continue

        if is_known_org(au):
            k = f"org_{canon_org(au)}_{y.lower()}"
            out.append(InTextCitation("author-year", m.group(0), k, year=y, surnames=(au,)))
        else:
            if looks_like_surname(au):
                out.append(InTextCitation("author-year", m.group(0), key_author_year(au, y), year=y, surnames=(au,)))

    # De-dup by raw
    uniq = []
    seen = set()
    for c in out:
        if c.raw not in seen:
            uniq.append(c)
            seen.add(c.raw)
    return uniq


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
    inside = inside.replace("–", "-")
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
    pat = re.compile(r"\(\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*\)")
    for m in pat.finditer(text or ""):
        inside = m.group(1)
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
def reconcile_author_year(cites: List[InTextCitation], refs: List[ReferenceEntry]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_key = defaultdict(list)
    for r in refs:
        ref_by_key[r.key].append(r)

    rows = []
    for c in cites:
        hits = ref_by_key.get(c.key, [])
        if not hits:
            rows.append({"in_text": c.raw, "status": "not_found", "matched_reference": ""})
        elif len(hits) == 1:
            rows.append({"in_text": c.raw, "status": "matched", "matched_reference": hits[0].raw})
        else:
            rows.append(
                {
                    "in_text": c.raw,
                    "status": f"ambiguous ({len(hits)})",
                    "matched_reference": " || ".join(h.raw[:200] for h in hits),
                }
            )
    df_c2r = pd.DataFrame(rows)

    cite_group = defaultdict(list)
    for c in cites:
        cite_group[c.key].append(c.raw)

    ref_rows = []
    for r in refs:
        cited_by = cite_group.get(r.key, [])
        ref_rows.append(
            {
                "reference": r.raw,
                "times_cited": len(cited_by),
                "cited_by": "; ".join(cited_by[:12]) + (" ..." if len(cited_by) > 12 else ""),
            }
        )
    df_r2c = pd.DataFrame(ref_rows).sort_values(["times_cited"], ascending=False)
    return df_c2r, df_r2c

def reconcile_numeric(cites: List[InTextCitation], refs: List[ReferenceEntry]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_key = {r.key: r for r in refs}

    rows = []
    for c in cites:
        r = ref_by_key.get(c.key)
        if not r:
            rows.append({"in_text": c.raw, "status": "not_found", "matched_reference": ""})
        else:
            rows.append({"in_text": c.raw, "status": "matched", "matched_reference": r.raw})
    df_c2r = pd.DataFrame(rows)

    cite_group = defaultdict(list)
    for c in cites:
        cite_group[c.key].append(c.raw)

    ref_rows = []
    for r in refs:
        cited_by = cite_group.get(r.key, [])
        ref_rows.append(
            {
                "reference": r.raw,
                "times_cited": len(cited_by),
                "cited_by": "; ".join(cited_by[:18]) + (" ..." if len(cited_by) > 18 else ""),
            }
        )
    df_r2c = pd.DataFrame(ref_rows).sort_values(["times_cited"], ascending=False)
    return df_c2r, df_r2c


# ============================
# Missing + Uncited (full raw)
# ============================
def build_missing_uncited(cites: List[InTextCitation], refs: List[ReferenceEntry]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    cite_keys = [c.key for c in cites]
    ref_keys = [r.key for r in refs]

    cite_count_by_raw = Counter([c.raw for c in cites])
    cite_key_by_raw = {}
    for c in cites:
        cite_key_by_raw.setdefault(c.raw, c.key)

    missing_rows = []
    ref_key_set = set(ref_keys)
    for raw, cnt in cite_count_by_raw.items():
        k = cite_key_by_raw.get(raw, "")
        if k and (k not in ref_key_set):
            missing_rows.append({"citation_in_text": raw, "count_in_text": cnt})

    if missing_rows:
        df_missing = pd.DataFrame(missing_rows).sort_values(
            ["count_in_text", "citation_in_text"], ascending=[False, True]
        )
    else:
        df_missing = pd.DataFrame(columns=["citation_in_text", "count_in_text"])

    cite_key_set = set(cite_keys)
    uncited_rows = [{"reference_full": r.raw} for r in refs if r.key not in cite_key_set]
    df_uncited = pd.DataFrame(uncited_rows) if uncited_rows else pd.DataFrame(columns=["reference_full"])

    summary = {
        "in_text_citations_found": len(cites),
        "reference_entries_found": len(refs),
        "missing_in_references": len(df_missing),
        "uncited_references": len(df_uncited),
    }
    return df_missing, df_uncited, summary


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
        lines = full.splitlines()
        idx = None
        for i, ln in enumerate(lines):
            s = ln.strip()
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

            merged = []
            buf = ""
            for ln in raw_lines:
                if not buf:
                    buf = ln
                    continue
                if re.match(r"^\s*(\[\d+\]|\d+[\.\)]|\d+\s)", ln):
                    merged.append(buf)
                    buf = ln
                else:
                    buf = (buf + " " + ln).strip()
            if buf:
                merged.append(buf)
            ref_lines = merged
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
            # Keep lines but also try to merge wrapped entries a bit
            lines = [x.strip() for x in pasted_refs.splitlines() if x.strip()]
            merged = []
            buf = ""
            for ln in lines:
                if not buf:
                    buf = ln
                    continue
                if re.match(r"^\s*(\[\d+\]|\d+[\.\)]|\d+\s)", ln):
                    merged.append(buf)
                    buf = ln
                else:
                    buf = (buf + " " + ln).strip()
            if buf:
                merged.append(buf)
            ref_lines = merged

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
        st.success("None detected.")
    else:
        st.dataframe(df_missing, use_container_width=True, height=420)
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
        st.dataframe(df_uncited, use_container_width=True, height=420)
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
    st.dataframe(df_c2r, use_container_width=True, height=420)
    st.download_button(
        "Download in-text to reference mapping (CSV)",
        df_c2r.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_intext_to_reference.csv",
        mime="text/csv",
    )

with right:
    st.markdown("### Reference → Cited by")
    st.dataframe(df_r2c, use_container_width=True, height=420)
    st.download_button(
        "Download reference to in-text mapping (CSV)",
        df_r2c.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_reference_to_intext.csv",
        mime="text/csv",
    )

with st.expander("Diagnostics"):
    st.markdown("#### Sample extracted in-text citations (first 80)")
    st.dataframe(pd.DataFrame([c.__dict__ for c in cites[:80]]), use_container_width=True)
    st.markdown("#### Sample parsed references (first 80)")
    st.dataframe(pd.DataFrame([r.__dict__ for r in refs[:80]]), use_container_width=True)
