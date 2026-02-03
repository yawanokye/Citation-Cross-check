import re
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
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


# -----------------------------
# Constants
# -----------------------------
YEAR = r"(?:19|20)\d{2}[a-z]?"
CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"

# Known organisation aliases (expand as needed)
ORG_ALIASES = {
    "unctad": ["unctad", "united nations conference on trade and development"],
    "who": ["who", "world health organization", "world health organisation"],
    "oecd": [
        "oecd",
        "organisation for economic co-operation and development",
        "organization for economic cooperation and development",
    ],
    "world bank": [
        "world bank",
        "international bank for reconstruction and development",
        "ibrd",
    ],
    "imf": ["imf", "international monetary fund"],
    "un": ["un", "united nations"],
    "unesco": ["unesco"],
    "unicef": ["unicef"],
}

ORG_ACRONYMS = {k.upper() for k in ["UNCTAD", "WHO", "OECD", "WORLD BANK", "IMF", "UN", "UNESCO", "UNICEF"]}


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class InTextCitation:
    style: str  # author-year | org-year | numeric
    raw: str
    key: str  # internal key (stable)
    pretty: str  # user-friendly display (e.g., "Kofi, 2000")
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


# -----------------------------
# Normalisation
# -----------------------------
def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_token(s: str) -> str:
    s = s.lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9\- ']", "", s)
    return norm_spaces(s)


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
    # "Saslavsky Shepherd" should NOT be treated as an organisation
    parts = name.strip().split()
    if len(parts) == 2 and all(p[:1].isupper() for p in parts):
        return True
    return False


# -----------------------------
# File readers
# -----------------------------
def read_docx(file) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)


def read_pdf(file) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber not installed")
    out = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            out.append(p.extract_text() or "")
    return "\n".join(out)


# -----------------------------
# Reference section detection
# -----------------------------
REF_HEADINGS = [
    r"^\s*references\s*$",
    r"^\s*reference\s*$",
    r"^\s*bibliography\s*$",
    r"^\s*works\s+cited\s*$",
    r"^\s*literature\s+cited\s*$",
]


def split_by_heading(text: str) -> Tuple[str, str, str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for pat in REF_HEADINGS:
            if re.match(pat, line.strip(), flags=re.IGNORECASE):
                main = "\n".join(lines[:i]).strip()
                refs = "\n".join(lines[i + 1 :]).strip()
                return main, refs, f"Found heading: {line.strip()}"
    return text.strip(), "", "No heading found"


def ref_line_score(line: str) -> int:
    s = line.strip()
    if not s:
        return 0

    if re.match(r"^\s*(\[\d+\]|\d+[\.\)])\s+", s):
        return 3

    if re.match(rf"^[A-Z][A-Za-z\-']+\s*,.*\(\s*{YEAR}\s*\)", s):
        return 4

    if re.match(rf"^[A-Z][A-Za-z\-']+\s*,.*\b{YEAR}\b", s):
        return 2

    if re.match(rf"^[A-Z][A-Z&\- ]{{2,}}\.\s*\(\s*{YEAR}\s*\)", s):
        return 4

    # org without dot: UNCTAD (2025) ...
    if re.match(rf"^[A-Z][A-Z&\- ]{{2,}}\s*\(\s*{YEAR}\s*\)", s):
        return 4

    return 0


def auto_detect_references_start(text: str) -> int:
    lines = text.splitlines()
    if len(lines) < 40:
        return max(0, int(len(lines) * 0.65))

    window = 30
    best_i = int(len(lines) * 0.70)
    best_score = -1

    start = int(len(lines) * 0.45)
    for i in range(start, len(lines) - window):
        score = sum(ref_line_score(lines[j]) for j in range(i, i + window))
        if score > best_score or (score == best_score and i > best_i):
            best_score = score
            best_i = i

    return best_i


# -----------------------------
# In-text citation extraction
# -----------------------------
def extract_author_year_citations(text: str) -> List[InTextCitation]:
    out: List[InTextCitation] = []

    # Narrative: Kofi (2000), Kofi & Mensah (2000), Kofi et al. (2000)
    narr = re.compile(
        rf"\b([A-Z][A-Za-z\-']+)(?:\s+(?:and|&)\s+([A-Z][A-Za-z\-']+)|\s+et\s+al\.)?\s*\(\s*({YEAR})\s*\)"
    )
    for m in narr.finditer(text):
        a1, y = m.group(1), m.group(3)
        if is_known_org(a1) and not looks_like_two_authors(a1):
            key = key_org_year(a1, y)
            pretty = f"{a1}, {y}"
            out.append(InTextCitation("org-year", m.group(0), key, pretty, a1, y))
        else:
            key = key_author_year(a1, y)
            pretty = f"{a1}, {y}"
            out.append(InTextCitation("author-year", m.group(0), key, pretty, a1, y))

    # Parenthetical blocks:
    # (Kofi, 2000; Mensah, 2001) or (Arvis et al., 2018) or (World Bank, 2023)
    paren_block = re.compile(rf"\(([^()]*?\b{YEAR}\b[^()]*)\)")
    for m in paren_block.finditer(text):
        block = m.group(1)
        parts = [p.strip() for p in block.split(";")]
        for p in parts:
            # Org: (UNCTAD, 2025) / (World Bank, 2023)
            org_seg = re.search(rf"^(.+?)\s*,\s*({YEAR})\b", p)
            if org_seg:
                cand, y = org_seg.group(1).strip(), org_seg.group(2)
                if is_known_org(cand) and not looks_like_two_authors(cand):
                    key = key_org_year(cand, y)
                    pretty = f"{cand}, {y}"
                    out.append(InTextCitation("org-year", f"({p})", key, pretty, cand, y))
                    continue

            # et al.: (Arvis et al., 2018)
            m_etal = re.search(rf"^([A-Z][A-Za-z\-']+)\s+et\s+al\.\s*,\s*({YEAR})\b", p)
            if m_etal:
                a1, y = m_etal.group(1), m_etal.group(2)
                key = key_author_year(a1, y)
                pretty = f"{a1}, {y}"
                out.append(InTextCitation("author-year", f"({p})", key, pretty, a1, y))
                continue

            # Multiple authors before one year: keep FIRST author only
            m_multi = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\b({YEAR})\b", p)
            if m_multi:
                first_author, y = m_multi.group(1), m_multi.group(2)
                if is_known_org(first_author) and not looks_like_two_authors(first_author):
                    key = key_org_year(first_author, y)
                    pretty = f"{first_author}, {y}"
                    out.append(InTextCitation("org-year", f"({p})", key, pretty, first_author, y))
                else:
                    key = key_author_year(first_author, y)
                    pretty = f"{first_author}, {y}"
                    out.append(InTextCitation("author-year", f"({p})", key, pretty, first_author, y))
                continue

            # Plain: (Kofi, 2000)
            m_plain = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,\s*({YEAR})\b", p)
            if m_plain:
                a1, y = m_plain.group(1), m_plain.group(2)
                key = key_author_year(a1, y)
                pretty = f"{a1}, {y}"
                out.append(InTextCitation("author-year", f"({p})", key, pretty, a1, y))
                continue

    # Organisation narrative: UNCTAD (2025), WHO (2020), World Bank (2023)
    org_narr = re.compile(rf"\b([A-Z][A-Za-z&.\- ]{{2,}}?)\s*\(\s*({YEAR})\s*\)")
    for m in org_narr.finditer(text):
        org, y = m.group(1).strip(), m.group(2)
        if looks_like_two_authors(org):
            continue
        if is_known_org(org):
            key = key_org_year(org, y)
            pretty = f"{org}, {y}"
            out.append(InTextCitation("org-year", m.group(0), key, pretty, org, y))

    return out


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


# -----------------------------
# Reference entry extraction
# -----------------------------
def split_reference_entries(ref_text: str) -> List[str]:
    ref_text = ref_text.strip()
    if not ref_text:
        return []

    chunks = [c.strip() for c in re.split(r"\n\s*\n", ref_text) if c.strip()]
    if len(chunks) >= 5:
        return chunks

    lines = [l.rstrip() for l in ref_text.splitlines() if l.strip()]
    entries = []
    buf = ""
    start_pat = re.compile(r"^(\[?\d+\]?[\.\)]\s+)|^([A-Z][A-Za-z\-']+\s*,)|^([A-Z][A-Z&\- ]{2,}\.?)")
    for line in lines:
        if start_pat.search(line) and buf:
            entries.append(buf.strip())
            buf = line
        else:
            buf = (buf + " " + line).strip() if buf else line
    if buf:
        entries.append(buf.strip())
    return entries


def parse_reference_author_year(entry: str) -> Optional[ReferenceEntry]:
    entry = entry.strip()

    # Organisation (with or without dot): UNCTAD (2025)... OR UNCTAD. (2025)...
    org_m = re.search(rf"^([A-Z][A-Z&\- ]{{2,}})\.?\s*\(\s*({YEAR})\s*\)", entry)
    if org_m:
        org, y = org_m.group(1).strip(), org_m.group(2)
        if is_known_org(org):
            return ReferenceEntry(entry, key_org_year(org, y), f"{org}, {y}", org, y)

    # Organisation (title case, with or without dot): World Bank (2023)...
    org2 = re.search(rf"^([A-Z][A-Za-z&.\- ]{{2,}}?)\.?\s*\(\s*({YEAR})\s*\)", entry)
    if org2:
        org, y = org2.group(1).strip(), org2.group(2)
        if is_known_org(org) and not looks_like_two_authors(org):
            return ReferenceEntry(entry, key_org_year(org, y), f"{org}, {y}", org, y)

    # Author-year: Surname, X. (2020). ...
    m = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\(\s*({YEAR})\s*\)", entry)
    if m:
        au, y = m.group(1), m.group(2)
        return ReferenceEntry(entry, key_author_year(au, y), f"{au}, {y}", au, y)

    # Harvard-ish: Surname, X. ... 2020 ...
    m2 = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\b({YEAR})\b", entry)
    if m2:
        au, y = m2.group(1), m2.group(2)
        return ReferenceEntry(entry, key_author_year(au, y), f"{au}, {y}", au, y)

    # Variant: "Korinek J ... (2011) ..."
    m_alt = re.search(rf"^([A-Z][A-Za-z\-']+)\s+[A-Z]\b.*?\(\s*({YEAR})\s*\)", entry)
    if m_alt:
        au, y = m_alt.group(1), m_alt.group(2)
        return ReferenceEntry(entry, key_author_year(au, y), f"{au}, {y}", au, y)

    return None


def parse_reference_numeric(entry: str) -> Optional[ReferenceEntry]:
    m = re.match(r"^\s*\[(\d+)\]\s*(.+)$", entry)
    if m:
        n = int(m.group(1))
        return ReferenceEntry(entry, key_numeric(n), f"[{n}]", number=n)

    m = re.match(r"^\s*(\d+)[\.\)]\s+(.+)$", entry)
    if m:
        n = int(m.group(1))
        return ReferenceEntry(entry, key_numeric(n), f"[{n}]", number=n)

    return None


# -----------------------------
# Year mismatch check
# -----------------------------
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
            rows.append(
                {
                    "author_or_org_key": b,
                    "years_in_text": ", ".join(sorted(cite_map[b])),
                    "years_in_references": ", ".join(sorted(ref_map[b])),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Online verification (Crossref + OpenAlex)
# -----------------------------
def extract_doi(text: str) -> Optional[str]:
    m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, flags=re.IGNORECASE)
    return m.group(1) if m else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def _get_json(url: str, params: dict) -> dict:
    r = requests.get(
        url,
        params=params,
        timeout=20,
        headers={"User-Agent": "citation-crosscheck/1.0"},
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def crossref_lookup_by_doi(doi: str) -> Optional[dict]:
    try:
        data = _get_json(f"{CROSSREF_API}/{doi}", params={})
        return data.get("message")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def crossref_search(query: str, rows: int = 5) -> List[dict]:
    data = _get_json(CROSSREF_API, params={"query.bibliographic": query, "rows": rows})
    return data.get("message", {}).get("items", []) or []


@st.cache_data(show_spinner=False)
def openalex_search(query: str, per_page: int = 5) -> List[dict]:
    data = _get_json(OPENALEX_API, params={"search": query, "per-page": per_page})
    return data.get("results", []) or []


def guess_year(ref: str) -> Optional[str]:
    m = re.search(rf"\b({YEAR})\b", ref)
    return m.group(1) if m else None


def guess_surname_or_org(ref: str) -> Optional[str]:
    m = re.match(r"^\s*([A-Z][A-Za-z\-']+)\s*,", ref)
    if m:
        return m.group(1)
    m2 = re.match(r"^\s*([A-Z][A-Z&\- ]{2,})\.?", ref)
    if m2:
        return m2.group(1)
    return None


def guess_title_snippet(ref: str) -> str:
    t = re.sub(rf"\(.*?{YEAR}.*?\)", " ", ref)
    t = re.sub(r"^[^\.]{1,120}\.\s*", " ", t)
    t = norm_spaces(t)
    words = t.split()
    return " ".join(words[:14])[:180]


def verify_reference_online(ref_entry: str, throttle_s: float = 0.2) -> dict:
    doi = extract_doi(ref_entry)
    y = guess_year(ref_entry) or ""
    who = guess_surname_or_org(ref_entry) or ""
    title_snip = guess_title_snippet(ref_entry)

    if doi:
        cr = crossref_lookup_by_doi(doi)
        if cr:
            return {"status": "verified", "source": "crossref", "doi": doi}

    query = " ".join([p for p in [who, y, title_snip] if p]).strip()
    if not query:
        query = ref_entry[:200]

    time.sleep(max(0.0, throttle_s))

    best = {"status": "not_found", "source": "", "score": 0, "doi": ""}

    try:
        items = crossref_search(query, rows=5)
        for it in items:
            title = (it.get("title") or [""])[0]
            issued = it.get("issued", {}).get("date-parts", [])
            year = str(issued[0][0]) if issued and issued[0] else ""
            authors = it.get("author") or []
            fam = authors[0].get("family") if authors else ""
            cand = f"{fam} {year} {title}"
            score = fuzz.WRatio(query, cand)
            if score > best["score"]:
                best = {
                    "status": "likely" if score >= 80 else "needs_review",
                    "source": "crossref",
                    "score": int(score),
                    "doi": it.get("DOI") or "",
                    "matched_year": year,
                    "matched_first_author": fam,
                    "matched_title": title[:180],
                }
    except Exception as e:
        best = {**best, "error_crossref": str(e)[:180]}

    try:
        items = openalex_search(query, per_page=5)
        for it in items:
            title = it.get("title") or ""
            year = str(it.get("publication_year") or "")
            authorships = it.get("authorships") or []
            fa = ""
            if authorships:
                dn = (authorships[0].get("author") or {}).get("display_name") or ""
                fa = dn.split()[-1] if dn else ""
            cand = f"{fa} {year} {title}"
            score = fuzz.WRatio(query, cand)
            if score > best.get("score", 0):
                doi2 = ""
                ids = it.get("ids") or {}
                if ids.get("doi"):
                    doi2 = ids["doi"].replace("https://doi.org/", "")
                best = {
                    "status": "likely" if score >= 80 else "needs_review",
                    "source": "openalex",
                    "score": int(score),
                    "doi": doi2,
                    "matched_year": year,
                    "matched_first_author": fa,
                    "matched_title": title[:180],
                }
    except Exception as e:
        best = {**best, "error_openalex": str(e)[:180]}

    if best["status"] == "not_found":
        best["score"] = 0

    best["query_used"] = query[:220]
    return best


# -----------------------------
# PDF report
# -----------------------------
def make_pdf_report(
    style_name: str,
    summary: dict,
    df_missing: pd.DataFrame,
    df_uncited: pd.DataFrame,
    df_mismatch: pd.DataFrame,
    df_verify: pd.DataFrame,
) -> bytes:
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

    def draw_table(title: str, df: pd.DataFrame, max_rows: int = 28):
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
            line = " | ".join(str(row[col])[:70] for col in cols)
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


# -----------------------------
# Streamlit UI
# -----------------------------
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

st.subheader("References detection")
st.write(ref_msg)

manual = st.checkbox(
    "Manually choose where References start (recommended if heading not found)",
    value=(ref_msg == "No heading found"),
)

if manual:
    lines = text.splitlines()
    guess = auto_detect_references_start(text)
    idx = st.slider("Select the line where the References section starts", 0, max(0, len(lines) - 1), guess)
    main_text = "\n".join(lines[:idx]).strip()
    ref_text = "\n".join(lines[idx:]).strip()

with st.expander("Preview detected References section"):
    st.text(ref_text[:4000] if ref_text else "No references detected yet")

if not ref_text.strip():
    st.warning("References section is empty. Paste your References or adjust the slider.")
    ref_text = st.text_area("Paste References section here", height=220)

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
suggestions = {}
if missing_keys and ref_keys:
    for mk in missing_keys:
        matches = process.extract(mk, ref_keys, scorer=fuzz.WRatio, limit=5)
        suggestions[mk] = [f"{m[0]}({int(m[1])})" for m in matches if m[1] >= 75]

key_to_pretty_cite = defaultdict(lambda: "")
for c in cites:
    key_to_pretty_cite[c.key] = c.pretty

key_to_pretty_ref = defaultdict(lambda: "")
for r in refs:
    key_to_pretty_ref[r.key] = r.pretty

df_missing = pd.DataFrame(
    [
        {
            "citation": key_to_pretty_cite.get(k, k),
            "count_in_text": cite_counts[k],
            "internal_key": k,
            "suggested_matches": ", ".join(suggestions.get(k, [])),
        }
        for k in missing_keys
    ]
)

df_uncited = pd.DataFrame(
    [
        {
            "reference": key_to_pretty_ref.get(k, k),
            "times_in_references": ref_counts[k],
            "internal_key": k,
        }
        for k in uncited_keys
    ]
)

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
    st.dataframe(df_missing, use_container_width=True)
    st.download_button(
        "Download missing (CSV)",
        df_missing.to_csv(index=False).encode("utf-8"),
        file_name="missing_in_references.csv",
        mime="text/csv",
    )

with c2:
    st.markdown("### In References but never cited")
    st.dataframe(df_uncited, use_container_width=True)
    st.download_button(
        "Download uncited (CSV)",
        df_uncited.to_csv(index=False).encode("utf-8"),
        file_name="uncited_references.csv",
        mime="text/csv",
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
            mime="text/csv",
        )
else:
    st.info("Year mismatch check applies to author–year styles only.")

# -----------------------------
# Online verification
# -----------------------------
st.divider()
st.subheader("Online verification (Crossref + OpenAlex)")

enable_verify = st.checkbox("Enable online verification", value=False)
throttle = st.slider("Throttle seconds between queries", 0.0, 2.0, 0.2, 0.1)

df_verify = pd.DataFrame()

if enable_verify:
    try:
        _ = _get_json("https://api.crossref.org/works", {"rows": 1})
        st.success("Crossref reachable.")
    except Exception as e:
        st.error("Online verification can’t reach Crossref from this deployment.")
        st.write(str(e))
        st.stop()

    rows = []
    progress = st.progress(0)
    total = len(refs) if refs else 1

    for i, r in enumerate(refs):
        res = verify_reference_online(r.raw, throttle_s=throttle)
        rows.append(
            {
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
            }
        )
        progress.progress(int((i + 1) / total * 100))

    df_verify = pd.DataFrame(rows)
    st.dataframe(df_verify, use_container_width=True)
    st.download_button(
        "Download verification (CSV)",
        df_verify.to_csv(index=False).encode("utf-8"),
        file_name="online_verification.csv",
        mime="text/csv",
    )

    flagged = df_verify[df_verify["status"].isin(["not_found", "needs_review"])]
    if len(flagged) > 0:
        st.warning(f"Flagged {len(flagged)} items as Not found or Needs review. Check these first.")

# -----------------------------
# PDF report
# -----------------------------
st.divider()
pdf_bytes = make_pdf_report(style, summary, df_missing, df_uncited, df_mismatch, df_verify)
st.download_button(
    "Download full PDF report",
    data=pdf_bytes,
    file_name="citation_crosscheck_report.pdf",
    mime="application/pdf",
)

# Debug tabs (optional)
with st.expander("Extracted items (debug)"):
    tab1, tab2 = st.tabs(["In-text citations", "Reference entries"])
    with tab1:
        st.dataframe(pd.DataFrame([c.__dict__ for c in cites]), use_container_width=True)
    with tab2:
        st.dataframe(pd.DataFrame([r.__dict__ for r in refs]), use_container_width=True)
