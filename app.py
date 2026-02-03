# ============================================================
# Citation Crosschecker App
# APA / Harvard / IEEE / Vancouver
# Offline consistency + Online verification (Crossref, OpenAlex)
# ============================================================

import re
import time
from io import BytesIO
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, List

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


# ============================================================
# Configuration
# ============================================================

YEAR = r"(?:19|20)\d{2}[a-z]?"

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"

ORG_ALIASES = {
    "world health organization": ["who", "world health organization"],
    "world bank": ["world bank", "international bank for reconstruction and development", "ibrd"],
    "oecd": ["oecd", "organisation for economic co-operation and development",
             "organization for economic cooperation and development"],
    "un": ["un", "united nations"],
    "unicef": ["unicef"],
    "unesco": ["unesco"],
    "imf": ["imf", "international monetary fund"],
}


# ============================================================
# Data structures
# ============================================================

@dataclass
class InTextCitation:
    style: str
    raw: str
    key: str
    author: Optional[str] = None
    year: Optional[str] = None
    number: Optional[int] = None


@dataclass
class ReferenceEntry:
    raw: str
    key: str
    author: Optional[str] = None
    year: Optional[str] = None
    number: Optional[int] = None


# ============================================================
# Utilities
# ============================================================

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_author(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\-’' ]", "", s)
    return norm_spaces(s.replace("’", "'"))


def canon_org(name: str) -> str:
    n = norm_author(name)
    for canon, variants in ORG_ALIASES.items():
        for v in variants:
            if n == norm_author(v):
                return canon
    return n


def key_author_year(author: str, year: str) -> str:
    return f"{norm_author(author)}_{year.lower()}"


def key_org_year(org: str, year: str) -> str:
    return f"org_{canon_org(org)}_{year.lower()}"


def key_numeric(n: int) -> str:
    return f"n_{n}"


# ============================================================
# File readers
# ============================================================

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


# ============================================================
# Split references
# ============================================================

def split_references(text: str):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(references|bibliography|works cited)\s*$", line, re.I):
            return "\n".join(lines[:i]), "\n".join(lines[i+1:]), "heading"
    return text, "", "none"


# ============================================================
# In-text citation extraction
# ============================================================

def extract_author_year_citations(text: str) -> List[InTextCitation]:
    out = []

    # Organisational narrative
    org_narr = re.compile(rf"\b([A-Z][A-Za-z&.\- ]{{2,}}?)\s*\(\s*({YEAR})\s*\)")
    for m in org_narr.finditer(text):
        org, y = m.group(1).strip(), m.group(2)
        if len(org.split()) >= 2 or org.upper() in {"WHO", "OECD", "IMF", "UNICEF", "UNESCO"}:
            out.append(InTextCitation(
                style="author-year-org",
                raw=m.group(0),
                key=key_org_year(org, y),
                author=org,
                year=y
            ))

    # Author narrative
    narr = re.compile(
        rf"\b([A-Z][A-Za-z\-']+)(?:\s+(?:and|&)\s+([A-Z][A-Za-z\-']+)|\s+et\s+al\.)?\s*\(\s*({YEAR})\s*\)"
    )
    for m in narr.finditer(text):
        a1, y = m.group(1), m.group(3)
        out.append(InTextCitation(
            style="author-year",
            raw=m.group(0),
            key=key_author_year(a1, y),
            author=a1,
            year=y
        ))

    # Parenthetical
    paren = re.compile(rf"\(([^()]*?\b{YEAR}\b[^()]*)\)")
    for m in paren.finditer(text):
        for part in m.group(1).split(";"):
            p = part.strip()

            org = re.search(rf"^([A-Z][A-Za-z&.\- ]+)\s*,\s*({YEAR})", p)
            if org:
                out.append(InTextCitation(
                    style="author-year-org",
                    raw=f"({p})",
                    key=key_org_year(org.group(1), org.group(2)),
                    author=org.group(1),
                    year=org.group(2)
                ))
                continue

            au = re.search(rf"([A-Z][A-Za-z\-']+).+?,\s*({YEAR})", p)
            if au:
                out.append(InTextCitation(
                    style="author-year",
                    raw=f"({p})",
                    key=key_author_year(au.group(1), au.group(2)),
                    author=au.group(1),
                    year=au.group(2)
                ))

    return out


def extract_numeric_citations(text: str) -> List[InTextCitation]:
    out = []
    pat = re.compile(r"\[(\d+(?:[-–]\d+)?(?:,\s*\d+(?:[-–]\d+)?)*)\]")
    for m in pat.finditer(text):
        for c in m.group(1).split(","):
            if "-" in c or "–" in c:
                a, b = re.split("[-–]", c)
                for n in range(int(a), int(b) + 1):
                    out.append(InTextCitation("numeric", m.group(0), key_numeric(n), number=n))
            else:
                n = int(c.strip())
                out.append(InTextCitation("numeric", m.group(0), key_numeric(n), number=n))
    return out


# ============================================================
# Reference parsing
# ============================================================

def split_reference_entries(ref_text: str):
    blocks = re.split(r"\n\s*\n", ref_text)
    if len(blocks) > 5:
        return [b.strip() for b in blocks if b.strip()]

    entries, buf = [], ""
    for l in ref_text.splitlines():
        if re.match(r"^\s*(\[\d+\]|\d+[\.\)])", l) and buf:
            entries.append(buf.strip())
            buf = l
        else:
            buf = buf + " " + l
    if buf.strip():
        entries.append(buf.strip())
    return entries


def parse_reference_author_year(entry: str) -> Optional[ReferenceEntry]:
    org = re.search(rf"^([A-Z][A-Za-z&.\- ]+)\.\s*\(\s*({YEAR})\)", entry)
    if org:
        return ReferenceEntry(entry, key_org_year(org.group(1), org.group(2)),
                              org.group(1), org.group(2))

    m = re.search(rf"^([A-Z][A-Za-z\-']+)\s*,.*?\(\s*({YEAR})\)", entry)
    if m:
        return ReferenceEntry(entry, key_author_year(m.group(1), m.group(2)),
                              m.group(1), m.group(2))
    return None


def parse_reference_numeric(entry: str) -> Optional[ReferenceEntry]:
    m = re.match(r"^\s*\[(\d+)\]", entry)
    if not m:
        m = re.match(r"^\s*(\d+)[\.\)]", entry)
    if m:
        return ReferenceEntry(entry, key_numeric(int(m.group(1))), number=int(m.group(1)))
    return None


# ============================================================
# Online verification
# ============================================================

def extract_doi(text: str):
    m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, re.I)
    return m.group(1) if m else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6))
def get_json(url, params):
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def verify_reference_online(ref: str) -> dict:
    doi = extract_doi(ref)
    if doi:
        try:
            d = get_json(f"{CROSSREF_API}/{doi}", {})
            return {"status": "verified", "doi": doi, "source": "crossref"}
        except Exception:
            pass

    q = ref[:250]
    try:
        cr = get_json(CROSSREF_API, {"query.bibliographic": q, "rows": 3})
        if cr.get("message", {}).get("items"):
            return {"status": "likely", "source": "crossref"}
    except Exception:
        pass

    try:
        oa = get_json(OPENALEX_API, {"search": q, "per-page": 3})
        if oa.get("results"):
            return {"status": "likely", "source": "openalex"}
    except Exception:
        pass

    return {"status": "not_found", "source": ""}


# ============================================================
# PDF report
# ============================================================

def make_pdf(summary, df_missing, df_uncited, df_verify):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    y = h - 2*cm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, y, "Citation Crosscheck Report")
    y -= 1*cm

    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        c.drawString(2*cm, y, f"{k.replace('_',' ').title()}: {v}")
        y -= 0.45*cm

    def table(title, df):
        nonlocal y
        c.showPage()
        y = h - 2*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, title)
        y -= 0.6*cm
        c.setFont("Helvetica", 9)
        if df.empty:
            c.drawString(2*cm, y, "None.")
            return
        for _, r in df.head(30).iterrows():
            c.drawString(2*cm, y, " | ".join(str(v)[:80] for v in r))
            y -= 0.4*cm
            if y < 2*cm:
                c.showPage()
                y = h - 2*cm

    table("Missing in References", df_missing)
    table("Uncited References", df_uncited)
    table("Online Verification", df_verify)

    c.save()
    return buf.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config("Citation Crosschecker", layout="wide")
st.title("Citation Crosschecker")

file = st.file_uploader("Upload DOCX / PDF / TXT", ["docx", "pdf", "txt"])
text = st.text_area("Or paste manuscript text", height=160)

style = st.selectbox("Citation style",
                     ["APA / Harvard (author–year)", "IEEE (numeric)", "Vancouver (numeric)"])

if file:
    if file.name.endswith(".docx"):
        text = read_docx(file)
    elif file.name.endswith(".pdf"):
        text = read_pdf(file)
    else:
        text = file.read().decode("utf-8", "ignore")

if not text.strip():
    st.stop()

main, ref, _ = split_references(text)
if not ref:
    ref = st.text_area("Paste References section", height=220)

if style.startswith("APA"):
    cites = extract_author_year_citations(main)
    refs = [parse_reference_author_year(r)
            for r in split_reference_entries(ref)]
    refs = [r for r in refs if r]
else:
    cites = extract_numeric_citations(main)
    refs = [parse_reference_numeric(r)
            for r in split_reference_entries(ref)]
    refs = [r for r in refs if r]

cite_keys = [c.key for c in cites]
ref_keys = [r.key for r in refs]

missing = sorted(set(cite_keys) - set(ref_keys))
uncited = sorted(set(ref_keys) - set(cite_keys))

df_missing = pd.DataFrame({"key": missing})
df_uncited = pd.DataFrame({"key": uncited})

summary = {
    "in_text_citations": len(cites),
    "reference_entries": len(refs),
    "missing_in_references": len(missing),
    "uncited_references": len(uncited),
}

st.subheader("Summary")
st.json(summary)

st.subheader("Missing in References")
st.dataframe(df_missing)

st.subheader("Uncited References")
st.dataframe(df_uncited)

# Online verification
st.subheader("Online verification")
if st.checkbox("Enable online verification"):
    rows = []
    for r in refs:
        res = verify_reference_online(r.raw)
        rows.append({"reference": r.raw[:200], **res})
        time.sleep(0.2)
    df_verify = pd.DataFrame(rows)
    st.dataframe(df_verify)

    pdf = make_pdf(summary, df_missing, df_uncited, df_verify)
    st.download_button("Download PDF report", pdf,
                       file_name="citation_report.pdf",
                       mime="application/pdf")
