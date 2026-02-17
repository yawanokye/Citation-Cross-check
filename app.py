# app.py
# Citation Cross-check (APA/Harvard + IEEE/Vancouver)
# - Extracts text from DOCX/PDF
# - Parses references section
# - Extracts in-text citations (author–year + numeric)
# - Reconciles: in-text → exact reference, and reference → cited by
# - Flags: cited-but-missing, in-refs-but-uncited
# - Optional online verification via Crossref + OpenAlex (lightweight + cached)

import re
import io
import time
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable, Set
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
PDFPLUMBER_OK = False

try:
    from docx import Document  # python-docx
    DOCX_OK = True
except Exception:
    DOCX_OK = False

try:
    import pdfplumber
    PDFPLUMBER_OK = True
except Exception:
    PDFPLUMBER_OK = False


# ----------------------------
# Data models
# ----------------------------
@dataclass
class InTextCitation:
    raw: str
    style: str  # "author_year" or "numeric"
    key: str
    year: Optional[str] = None
    surnames: Optional[List[str]] = None
    nums: Optional[List[int]] = None


@dataclass
class ReferenceEntry:
    raw: str
    style: str  # "author_year" or "numeric"
    key: str
    keys_all: List[str]
    pretty: str
    year: Optional[str] = None
    author_or_org: Optional[str] = None
    number: Optional[int] = None


# ----------------------------
# Helpers: normalization
# ----------------------------
def norm_space(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s


def norm_key(s: str) -> str:
    s = norm_unicode(s).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def titleish(s: str) -> str:
    s = norm_space(s)
    if not s:
        return ""
    return s[0].upper() + s[1:]


def looks_like_year_only_parenthetical(raw: str) -> bool:
    # Ex: "(1970)" should NOT be treated as a citation (it caused false positives)
    return bool(re.fullmatch(r"\(\s*\d{4}[a-z]?\s*\)", raw.strip()))


def looks_like_sentence_fragment(raw: str) -> bool:
    # Stop the app from treating random phrases as citations
    # e.g. "Traditional methods—inclu..." caused by over-greedy parsing
    raw = raw.strip()
    if len(raw) < 6:
        return True
    # If it has no digits and no comma and no bracket/paren patterns, it's likely not a cite
    if not re.search(r"\d", raw) and not re.search(r"[\[\]\(\),;]", raw):
        return True
    return False


# ----------------------------
# Text extraction
# ----------------------------
def extract_docx_text(file_bytes: bytes) -> str:
    if not DOCX_OK:
        raise RuntimeError("python-docx is not installed.")
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)

    parts = []
    # Paragraphs
    for p in doc.paragraphs:
        t = p.text or ""
        t = t.strip()
        if t:
            parts.append(t)

    # Tables (sometimes references end up in tables)
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                t = (cell.text or "").strip()
                if t:
                    parts.append(t)

    return "\n".join(parts)


def extract_pdf_text(file_bytes: bytes) -> str:
    if not PDFPLUMBER_OK:
        raise RuntimeError("pdfplumber is not installed.")
    bio = io.BytesIO(file_bytes)
    out = []
    with pdfplumber.open(bio) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            t = t.strip()
            if t:
                out.append(t)
    return "\n".join(out)


# ----------------------------
# Split main text vs references
# ----------------------------
REF_HEADINGS = [
    r"references",
    r"bibliography",
    r"works\s+cited",
    r"reference\s+list",
    r"literature\s+cited",
]

def split_main_and_references(full_text: str) -> Tuple[str, str]:
    text = norm_unicode(full_text)
    lines = text.splitlines()

    # Find the last "References" heading (papers sometimes mention "references" earlier)
    ref_idx = None
    for i, line in enumerate(lines):
        l = norm_space(line).lower()
        if any(re.fullmatch(rf"{h}\.?\s*", l) for h in REF_HEADINGS):
            ref_idx = i

    if ref_idx is None:
        # Fallback: "REFERENCES" in caps, or line starts with "References"
        for i, line in enumerate(lines):
            l = norm_space(line)
            if re.fullmatch(r"(REFERENCES|BIBLIOGRAPHY|WORKS CITED)\.?\s*", l):
                ref_idx = i

    if ref_idx is None:
        # No explicit references heading found
        return text, ""

    main = "\n".join(lines[:ref_idx]).strip()
    refs = "\n".join(lines[ref_idx + 1 :]).strip()
    return main, refs


# ----------------------------
# Reference parsing (numeric + author-year)
# ----------------------------
def is_probably_numeric_refs(ref_text: str) -> bool:
    # If references include many lines starting with [n] or n. patterns, treat as numeric
    lines = [norm_space(x) for x in ref_text.splitlines() if norm_space(x)]
    if not lines:
        return False
    score = 0
    for ln in lines[:120]:
        if re.match(r"^\[\s*\d{1,4}\s*\]", ln):
            score += 2
        if re.match(r"^\d{1,4}\.\s+", ln):
            score += 1
    return score >= 6


def chunk_reference_lines(ref_text: str) -> List[str]:
    # Join wrapped lines into entries.
    # Heuristic: new entry begins with:
    # - [n]
    # - n.
    # - AuthorSurname, Initial.
    # - Organization name, or uppercase-ish start + year later
    lines = [x.rstrip() for x in ref_text.splitlines()]
    lines = [ln for ln in lines if norm_space(ln)]

    entries = []
    cur = ""

    def starts_new(ln: str) -> bool:
        l = norm_space(ln)
        if re.match(r"^\[\s*\d{1,4}\s*\]", l):
            return True
        if re.match(r"^\d{1,4}\.\s+", l):
            return True
        # Author-year ref lines commonly start with "Surname," or "Surname, X."
        if re.match(r"^[A-Z][A-Za-z'’\-]+,\s*[A-Z]", l):
            return True
        # Some styles: "Surname, A., Surname, B., & Surname, C."
        if re.match(r"^[A-Z][A-Za-z'’\-]+,\s*[A-Z]\.", l):
            return True
        # Organization start: "World Health Organization."
        if re.match(r"^[A-Z][A-Za-z&.\- ]{3,}\.\s*\(\d{4}[a-z]?\)", l):
            return True
        return False

    for ln in lines:
        if not cur:
            cur = ln
            continue
        if starts_new(ln):
            entries.append(norm_space(cur))
            cur = ln
        else:
            cur = cur + " " + ln

    if cur:
        entries.append(norm_space(cur))
    return entries


def split_authors_string(auth_str: str) -> List[str]:
    # Handles "Krejcie and Morgan", "Bartlett, Kotrlik, and Higgins", "A & B", "A, B, & C"
    s = norm_space(auth_str)
    # Remove "et al."
    s = re.sub(r"\bet\s+al\.?\b", "", s, flags=re.I).strip()

    # Convert separators to commas
    s = re.sub(r"\s*&\s*", ", ", s)
    s = re.sub(r"\s+and\s+", ", ", s, flags=re.I)
    s = re.sub(r",\s*and\s+", ", ", s, flags=re.I)

    parts = [p.strip() for p in s.split(",") if p.strip()]
    surnames = []
    for p in parts:
        # If "Surname, A." style, take surname before comma
        # Here p is already split by commas, so it may be "Surname" or "Surname A."
        # Keep first token that starts with a letter
        token = p.split()[0].strip()
        token = re.sub(r"[^A-Za-z'’\-]", "", token)
        if token:
            surnames.append(token)
    return surnames


KNOWN_ORGS = {
    "world health organization",
    "who",
    "united nations",
    "world bank",
    "international monetary fund",
    "imf",
    "oecd",
    "unicef",
    "unesco",
}

def is_known_org(name: str) -> bool:
    n = norm_key(name)
    n = n.replace("_", " ")
    return n in KNOWN_ORGS


def looks_like_two_authors(org_candidate: str) -> bool:
    # Avoid misclassifying "Krejcie and Morgan" as an org
    s = org_candidate.strip()
    if re.search(r"\b(and|&)\b", s, flags=re.I):
        # likely authors
        return True
    return False


def key_author_year_surnames(surnames: List[str], year: str) -> Tuple[str, str]:
    # returns (au_first, au2_first_two) keys
    y = norm_key(year)
    if not surnames:
        return f"au__{y}", f"au2__{y}"
    s1 = norm_key(surnames[0])
    au = f"au_{s1}_{y}"
    if len(surnames) >= 2:
        s2 = norm_key(surnames[1])
        au2 = f"au2_{s1}_{s2}_{y}"
    else:
        au2 = f"au2_{s1}_{y}"
    return au, au2


def key_org_year(org: str, year: str) -> str:
    return f"org_{norm_key(org)}_{norm_key(year)}"


def key_num(n: int) -> str:
    return f"n_{int(n)}"


def parse_reference_entry_author_year(raw: str) -> ReferenceEntry:
    r = norm_space(raw)

    # Numeric prefix in an author-year list (rare but happens)
    r = re.sub(r"^\[\s*\d{1,4}\s*\]\s*", "", r)
    r = re.sub(r"^\d{1,4}\.\s+", "", r)

    # Year in parentheses or after dot
    m_year = re.search(r"\(\s*(\d{4}[a-z]?)\s*\)", r)
    year = m_year.group(1) if m_year else None

    # Authors segment: from start until first "("year")" or first ". (year)" etc.
    auth_seg = r
    if m_year:
        auth_seg = r[: m_year.start()].strip()
    else:
        # Try: "Author. 2001." (not common here)
        m_year2 = re.search(r"\b(\d{4}[a-z]?)\b", r)
        if m_year2:
            year = m_year2.group(1)
            auth_seg = r[: m_year2.start()].strip()

    # Determine org vs authors
    author_or_org = auth_seg.strip(" .")
    surnames = split_authors_string(author_or_org)

    keys_all: List[str] = []
    primary_key = ""

    if year:
        # Org form "World Health Organization"
        if is_known_org(author_or_org) and not looks_like_two_authors(author_or_org):
            primary_key = key_org_year(author_or_org, year)
            keys_all = [primary_key]
        else:
            au, au2 = key_author_year_surnames(surnames, year)
            primary_key = au2 if len(surnames) >= 2 else au
            # include both keys for matching
            keys_all = list(dict.fromkeys([au, au2]))
    else:
        # No year found: use first author as key (weak)
        primary_key = f"au_{norm_key(surnames[0] if surnames else author_or_org)}_noyear"
        keys_all = [primary_key]

    pretty = ""
    if year:
        if is_known_org(author_or_org) and not looks_like_two_authors(author_or_org):
            pretty = f"{titleish(author_or_org)} {year}"
        else:
            pretty = f"{titleish(surnames[0] if surnames else author_or_org)} {year}"
    else:
        pretty = titleish(surnames[0] if surnames else author_or_org)

    return ReferenceEntry(
        raw=r,
        style="author_year",
        key=primary_key,
        keys_all=keys_all,
        pretty=pretty,
        year=year,
        author_or_org=author_or_org,
        number=None,
    )


def parse_reference_entry_numeric(raw: str) -> ReferenceEntry:
    r = norm_space(raw)
    m = re.match(r"^\[\s*(\d{1,4})\s*\]\s*(.+)$", r)
    num = None
    body = r
    if m:
        num = int(m.group(1))
        body = m.group(2).strip()
    else:
        m2 = re.match(r"^(\d{1,4})\.\s+(.+)$", r)
        if m2:
            num = int(m2.group(1))
            body = m2.group(2).strip()

    if num is None:
        # If unnumbered, assign None key; it will be "uncitable" in numeric mapping
        k = "n__"
        keys_all = [k]
        pretty = body[:80]
    else:
        k = key_num(num)
        keys_all = [k]
        pretty = f"[{num}] {body[:70]}"

    return ReferenceEntry(
        raw=r,
        style="numeric",
        key=k,
        keys_all=keys_all,
        pretty=pretty,
        year=None,
        author_or_org=None,
        number=num,
    )


def parse_references(ref_text: str, prefer_style: str) -> List[ReferenceEntry]:
    if not ref_text.strip():
        return []

    chunks = chunk_reference_lines(ref_text)

    # Decide numeric vs author-year
    numeric = False
    if prefer_style in ("IEEE/Vancouver (numeric)",):
        numeric = True
    elif prefer_style in ("APA/Harvard (author–year)",):
        numeric = False
    else:
        numeric = is_probably_numeric_refs(ref_text)

    refs: List[ReferenceEntry] = []
    for c in chunks:
        if numeric:
            refs.append(parse_reference_entry_numeric(c))
        else:
            refs.append(parse_reference_entry_author_year(c))

    return refs


# ----------------------------
# In-text citation extraction
# ----------------------------
def extract_author_year_parenthetical(main_text: str) -> List[InTextCitation]:
    text = norm_unicode(main_text)

    # Find parentheses blocks that look like author–year citations.
    # REQUIRE: at least one letter and a 4-digit year inside.
    paren_blocks = re.findall(r"\(([^)]*?\d{4}[a-z]?(?:[^)]*?))\)", text)

    cites: List[InTextCitation] = []

    for block in paren_blocks:
        raw_block = "(" + block + ")"
        if looks_like_year_only_parenthetical(raw_block):
            # prevents (1970) etc.
            continue

        if not re.search(r"[A-Za-z]", block):
            continue

        # Split multiple citations by semicolons
        segments = [seg.strip() for seg in re.split(r"\s*;\s*", block) if seg.strip()]

        for seg in segments:
            # Remove page locators
            seg_clean = re.sub(r",?\s*(p|pp)\.?\s*\d+([–-]\d+)?", "", seg, flags=re.I).strip()

            # Typical patterns:
            # "Alagidede, Tweneboah, & Adam, 2008"
            # "Kofi et al., 2020"
            # "Button et al., 2013"
            # "Krejcie & Morgan, 1970"
            m = re.search(r"^(?P<auth>.+?)\s*,\s*(?P<year>\d{4}[a-z]?)\s*$", seg_clean)
            if not m:
                # Sometimes: "Author (Year)" appears inside parentheses as "Author, Year:..."
                m = re.search(r"^(?P<auth>.+?)\s+(?P<year>\d{4}[a-z]?)\s*$", seg_clean)
            if not m:
                continue

            auth = norm_space(m.group("auth"))
            year = m.group("year")
            # auth may include "&" and commas; take surnames list
            surnames = split_authors_string(auth)

            au, au2 = key_author_year_surnames(surnames, year)
            # Prefer au2 if we have 2+ authors; this stops "Morgan (1970)" collapsing cases
            key = au2 if len(surnames) >= 2 else au

            cites.append(
                InTextCitation(
                    raw=f"({seg.strip()})",
                    style="author_year",
                    key=key,
                    year=year,
                    surnames=surnames,
                    nums=None,
                )
            )

    return cites


def extract_author_year_narrative(main_text: str) -> List[InTextCitation]:
    text = norm_unicode(main_text)

    # Narrative patterns:
    # "Krejcie and Morgan (1970)"
    # "Bartlett, Kotrlik, and Higgins (2001)"
    # "Kofi et al. (2020)"
    # Capture author list BEFORE "("year")"
    pattern = re.compile(
        r"""
        (?P<auth>
            (?:[A-Z][A-Za-z'’\-]+)                                  # first surname
            (?:
                (?:\s+et\s+al\.?)                                  # et al
                |
                (?:\s*(?:,)\s*[A-Z][A-Za-z'’\-]+)*                 # , Surname, Surname
                (?:\s*(?:,)?\s*(?:and|&)\s*[A-Z][A-Za-z'’\-]+)?    # and Surname
            )?
        )
        \s*\(\s*(?P<year>\d{4}[a-z]?)\s*\)
        """,
        re.VERBOSE,
    )

    cites: List[InTextCitation] = []
    for m in pattern.finditer(text):
        auth = norm_space(m.group("auth"))
        year = m.group("year")

        # Guard: don't treat "Figure 2 (2020)" as narrative citation
        if re.search(r"\b(figure|table|equation|section|appendix)\b", auth, flags=re.I):
            continue

        surnames = split_authors_string(auth)
        if not surnames:
            continue

        au, au2 = key_author_year_surnames(surnames, year)
        key = au2 if len(surnames) >= 2 else au

        cites.append(
            InTextCitation(
                raw=f"{auth} ({year})",
                style="author_year",
                key=key,
                year=year,
                surnames=surnames,
                nums=None,
            )
        )
    return cites


def extract_numeric_citations(main_text: str) -> List[InTextCitation]:
    text = norm_unicode(main_text)

    cites: List[InTextCitation] = []

    # IEEE-style: [1], [1, 2, 3], [1–3], [1-3]
    bracket_pat = re.compile(
        r"""
        \[
            \s*
            (?:
                \d{1,4}
                (?:\s*[-–]\s*\d{1,4}|\s*,\s*\d{1,4}|\s+\d{1,4})*
            )
            \s*
        \]
        """,
        re.VERBOSE,
    )

    for m in bracket_pat.finditer(text):
        raw = m.group(0)
        nums = parse_numeric_cite_numbers(raw)
        if not nums:
            continue
        # Create one InTextCitation per bracket block (key = joined)
        key = "nset_" + "_".join(str(n) for n in nums)
        cites.append(InTextCitation(raw=raw, style="numeric", key=key, nums=nums))

    # Vancouver parentheses numeric: (1), (1,2), (1–3)
    # IMPORTANT: avoid years like (1970) by restricting 1–3 digits
    paren_num_pat = re.compile(
        r"""
        \(
            \s*
            (?:
                \d{1,3}
                (?:\s*[-–]\s*\d{1,3}|\s*,\s*\d{1,3})*
            )
            \s*
        \)
        """,
        re.VERBOSE,
    )

    for m in paren_num_pat.finditer(text):
        raw = m.group(0)
        # extra guard: don't capture (pp. 12) etc
        if re.search(r"[A-Za-z]", raw):
            continue
        nums = parse_numeric_cite_numbers(raw)
        if not nums:
            continue
        key = "nset_" + "_".join(str(n) for n in nums)
        cites.append(InTextCitation(raw=raw, style="numeric", key=key, nums=nums))

    return cites


def parse_numeric_cite_numbers(raw: str) -> List[int]:
    s = raw.strip().strip("[]()")
    s = s.replace("–", "-")
    s = re.sub(r"\s+", "", s)
    if not s:
        return []
    nums: List[int] = []
    parts = s.split(",")
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            if a.isdigit() and b.isdigit():
                ia = int(a)
                ib = int(b)
                if ia <= ib and (ib - ia) <= 2000:
                    nums.extend(list(range(ia, ib + 1)))
        else:
            if p.isdigit():
                nums.append(int(p))
    # de-dupe while preserving order
    seen = set()
    out = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def extract_in_text_citations(main_text: str, prefer_style: str) -> List[InTextCitation]:
    if prefer_style == "IEEE/Vancouver (numeric)":
        return extract_numeric_citations(main_text)

    # author–year
    cites = []
    cites.extend(extract_author_year_parenthetical(main_text))
    cites.extend(extract_author_year_narrative(main_text))
    return cites


# ----------------------------
# Reconciliation + missing/uncited
# ----------------------------
def build_ref_index(refs: List[ReferenceEntry]) -> Dict[str, List[ReferenceEntry]]:
    ref_by_key = defaultdict(list)
    for r in refs:
        for k in r.keys_all:
            ref_by_key[k].append(r)
        # also index primary
        ref_by_key[r.key].append(r)
    return ref_by_key


def reconcile_author_year(
    cites: List[InTextCitation], refs: List[ReferenceEntry]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_key = build_ref_index(refs)

    rows = []
    matched_ref_keys: List[str] = []

    for c in cites:
        hits = ref_by_key.get(c.key, [])

        # fallback: if cite is au2_..., try au_... (first-author-only)
        if not hits and c.key.startswith("au2_"):
            parts = c.key.split("_")
            # au2_s1_s2_year -> au_s1_year
            if len(parts) >= 4:
                fb = f"au_{parts[1]}_{parts[-1]}"
                hits = ref_by_key.get(fb, [])

        if not hits:
            rows.append({"in_text": c.raw, "status": "not_found", "matched_reference": "", "matched_key": ""})
        elif len(hits) == 1:
            matched_ref_keys.append(hits[0].key)
            rows.append(
                {"in_text": c.raw, "status": "matched", "matched_reference": hits[0].raw, "matched_key": hits[0].key}
            )
        else:
            # multiple refs share the same key
            matched_ref_keys.append(hits[0].key)
            rows.append(
                {
                    "in_text": c.raw,
                    "status": f"ambiguous ({len(hits)})",
                    "matched_reference": " || ".join(h.raw[:180] for h in hits),
                    "matched_key": hits[0].key,
                }
            )

    df_c2r = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["in_text", "status", "matched_reference", "matched_key"])

    # reverse mapping: group by matched_reference (by key)
    cite_group = defaultdict(list)
    for row in rows:
        k = row.get("matched_key", "")
        if k:
            cite_group[k].append(row["in_text"])

    ref_rows = []
    for r in refs:
        cited_by = cite_group.get(r.key, [])
        ref_rows.append(
            {
                "reference": r.raw,
                "times_cited": int(len(cited_by)),
                "cited_by": "; ".join(cited_by[:12]) + (" ..." if len(cited_by) > 12 else ""),
                "ref_key": r.key,
            }
        )

    df_r2c = pd.DataFrame(ref_rows) if ref_rows else pd.DataFrame(columns=["reference", "times_cited", "cited_by", "ref_key"])
    # Always safe sort
    if "times_cited" not in df_r2c.columns:
        df_r2c["times_cited"] = 0
    if len(df_r2c) > 0:
        df_r2c = df_r2c.sort_values(["times_cited", "reference"], ascending=[False, True]).reset_index(drop=True)

    return df_c2r, df_r2c


def reconcile_numeric(
    cites: List[InTextCitation], refs: List[ReferenceEntry]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Map numbers → references
    ref_by_num: Dict[int, ReferenceEntry] = {}
    for r in refs:
        if r.number is not None:
            ref_by_num[int(r.number)] = r

    rows = []
    cite_to_nums = []
    for c in cites:
        nums = c.nums or []
        for n in nums:
            r = ref_by_num.get(n)
            if not r:
                rows.append({"in_text": c.raw, "number": n, "status": "not_found", "matched_reference": ""})
            else:
                rows.append({"in_text": c.raw, "number": n, "status": "matched", "matched_reference": r.raw})
                cite_to_nums.append(n)

    df_c2r = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["in_text", "number", "status", "matched_reference"])

    # reverse mapping
    cite_count = Counter(cite_to_nums)
    ref_rows = []
    for r in refs:
        if r.number is None:
            continue
        n = int(r.number)
        ref_rows.append(
            {
                "reference": r.raw,
                "number": n,
                "times_cited": int(cite_count.get(n, 0)),
            }
        )

    df_r2c = pd.DataFrame(ref_rows) if ref_rows else pd.DataFrame(columns=["reference", "number", "times_cited"])
    if "times_cited" not in df_r2c.columns:
        df_r2c["times_cited"] = 0
    if len(df_r2c) > 0:
        df_r2c = df_r2c.sort_values(["times_cited", "number"], ascending=[False, True]).reset_index(drop=True)

    return df_c2r, df_r2c


def missing_and_uncited_author_year(
    df_c2r: pd.DataFrame, refs: List[ReferenceEntry]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Missing: in_text where status not_found
    df_missing = df_c2r[df_c2r["status"].str.startswith("not_found")].copy() if len(df_c2r) else pd.DataFrame(
        columns=["citation_in_text", "count_in_text"]
    )
    if len(df_missing):
        df_missing = (
            df_missing.groupby("in_text")
            .size()
            .reset_index(name="count_in_text")
            .rename(columns={"in_text": "citation_in_text"})
        )
    else:
        df_missing = pd.DataFrame(columns=["citation_in_text", "count_in_text"])

    # Uncited: references with times_cited = 0
    # We compute from df_r2c instead in the UI, but here keep generic
    # This function expects refs and df_c2r only, so infer cited keys from df_c2r
    cited_keys = set(df_c2r.loc[df_c2r["matched_key"].astype(str) != "", "matched_key"].astype(str)) if len(df_c2r) else set()
    uncited_rows = [{"reference_full": r.raw} for r in refs if r.key not in cited_keys]
    df_uncited = pd.DataFrame(uncited_rows) if uncited_rows else pd.DataFrame(columns=["reference_full"])

    return df_missing, df_uncited


def missing_and_uncited_numeric(
    df_c2r: pd.DataFrame, refs: List[ReferenceEntry]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Missing: numbers cited but not found
    if len(df_c2r):
        miss = df_c2r[df_c2r["status"] == "not_found"].copy()
        if len(miss):
            df_missing = (
                miss.groupby(["number"])
                .size()
                .reset_index(name="count_in_text")
                .assign(citation_in_text=lambda d: d["number"].apply(lambda x: f"[{x}]"))
                .loc[:, ["citation_in_text", "count_in_text", "number"]]
            )
        else:
            df_missing = pd.DataFrame(columns=["citation_in_text", "count_in_text", "number"])
    else:
        df_missing = pd.DataFrame(columns=["citation_in_text", "count_in_text", "number"])

    cited_nums = set(df_c2r.loc[df_c2r["status"] == "matched", "number"].astype(int)) if len(df_c2r) else set()
    uncited_rows = []
    for r in refs:
        if r.number is None:
            continue
        if int(r.number) not in cited_nums:
            uncited_rows.append({"reference_full": r.raw, "number": int(r.number)})
    df_uncited = pd.DataFrame(uncited_rows) if uncited_rows else pd.DataFrame(columns=["reference_full", "number"])
    if len(df_uncited):
        df_uncited = df_uncited.sort_values("number").reset_index(drop=True)

    return df_missing, df_uncited


def suggest_matches_for_missing(
    missing_cites: Iterable[str], refs: List[ReferenceEntry], top_k: int = 3
) -> Dict[str, str]:
    ref_texts = [r.raw for r in refs]
    out = {}
    for c in missing_cites:
        scores = []
        for rr in ref_texts:
            scores.append((fuzz.token_set_ratio(c, rr), rr))
        scores.sort(key=lambda x: x[0], reverse=True)
        sug = " || ".join([s[1][:180] for s in scores[:top_k]]) if scores else ""
        out[c] = sug
    return out


# ----------------------------
# Online verification (Crossref + OpenAlex)
# ----------------------------
CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_URL = "https://api.openalex.org/works"

def build_biblio_query(ref_raw: str) -> str:
    # Light query string from reference:
    # keep author-ish + year + title-ish substring
    r = ref_raw
    r = re.sub(r"^\[\s*\d{1,4}\s*\]\s*", "", r)
    r = re.sub(r"^\d{1,4}\.\s+", "", r)
    r = re.sub(r"\s+", " ", r).strip()
    # Cut long refs
    return r[:280]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def crossref_lookup(query: str, timeout: int = 15) -> Optional[dict]:
    params = {"query.bibliographic": query, "rows": 1}
    r = requests.get(CROSSREF_URL, params=params, timeout=timeout, headers={"User-Agent": "citation-cross-check/1.0"})
    r.raise_for_status()
    data = r.json()
    items = (data.get("message") or {}).get("items") or []
    return items[0] if items else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
def openalex_lookup(query: str, timeout: int = 15) -> Optional[dict]:
    params = {"search": query, "per-page": 1}
    r = requests.get(OPENALEX_URL, params=params, timeout=timeout, headers={"User-Agent": "citation-cross-check/1.0"})
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    return results[0] if results else None


def score_match(ref_raw: str, title: str, year: Optional[str]) -> int:
    base = fuzz.token_set_ratio(ref_raw, title or "")
    if year and title and year in ref_raw:
        base = min(100, base + 5)
    return int(base)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def verify_reference_online(ref_raw: str) -> Dict[str, str]:
    q = build_biblio_query(ref_raw)

    cr = None
    oa = None
    cr_title = ""
    cr_year = ""
    cr_doi = ""
    cr_url = ""

    oa_title = ""
    oa_year = ""
    oa_doi = ""
    oa_url = ""

    try:
        cr = crossref_lookup(q)
    except Exception:
        cr = None

    if cr:
        cr_title = ((cr.get("title") or [""])[0] or "").strip()
        cr_year = str(((cr.get("issued") or {}).get("date-parts") or [[None]])[0][0] or "").strip()
        cr_doi = (cr.get("DOI") or "").strip()
        cr_url = (cr.get("URL") or "").strip()

    try:
        oa = openalex_lookup(q)
    except Exception:
        oa = None

    if oa:
        oa_title = (oa.get("title") or "").strip()
        oa_year = str(oa.get("publication_year") or "").strip()
        oa_doi = (oa.get("doi") or "").replace("https://doi.org/", "").strip()
        oa_url = (oa.get("id") or "").strip()

    best_title = cr_title or oa_title
    best_year = cr_year or oa_year
    best_doi = cr_doi or oa_doi
    best_src = "Crossref" if cr_title else ("OpenAlex" if oa_title else "")

    match_score = score_match(ref_raw, best_title, best_year) if best_title else 0

    return {
        "source": best_src,
        "match_score": str(match_score),
        "title": best_title,
        "year": best_year,
        "doi": best_doi,
        "url": cr_url or oa_url,
    }


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Citation Cross-check", layout="wide")
st.title("Citation Cross-check")
st.caption("Checks in-text citations vs reference list, with reconciliation tables. Supports APA/Harvard and IEEE/Vancouver.")

with st.sidebar:
    st.subheader("Settings")

    style = st.selectbox(
        "Citation style",
        ["APA/Harvard (author–year)", "IEEE/Vancouver (numeric)"],
        index=0,
    )

    show_debug = st.checkbox("Show debug tables", value=False)

    st.divider()
    st.subheader("Online verification (optional)")
    enable_verify = st.checkbox("Verify references online (Crossref + OpenAlex)", value=False)
    max_verify = st.slider("Max references to verify", min_value=5, max_value=120, value=25, step=5)
    st.caption("Tip: keep this low for speed and to avoid rate limits.")

uploaded = st.file_uploader("Upload a manuscript (DOCX or PDF)", type=["docx", "pdf"])

if not uploaded:
    st.stop()

file_bytes = uploaded.read()

# Extract text
with st.spinner("Reading file..."):
    try:
        if uploaded.name.lower().endswith(".docx"):
            full_text = extract_docx_text(file_bytes)
        else:
            full_text = extract_pdf_text(file_bytes)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

full_text = norm_unicode(full_text)

main_text, ref_text = split_main_and_references(full_text)

if not ref_text.strip():
    st.warning("I couldn't find a References section heading. The app will still extract in-text citations, but reference matching will be limited.")

# Parse
with st.spinner("Parsing references and in-text citations..."):
    refs = parse_references(ref_text, style) if ref_text else []
    cites = extract_in_text_citations(main_text, style)

# Remove obvious garbage citations (extra safety)
clean_cites = []
for c in cites:
    if looks_like_sentence_fragment(c.raw):
        continue
    clean_cites.append(c)
cites = clean_cites

# Counts
cite_counter = Counter([c.raw for c in cites])

# Reconcile
with st.spinner("Reconciling citations ↔ references..."):
    if style.startswith("APA/Harvard"):
        df_c2r, df_r2c = reconcile_author_year(cites, refs)
        df_missing, df_uncited = missing_and_uncited_author_year(df_c2r, refs)
        if len(df_missing):
            sugg = suggest_matches_for_missing(df_missing["citation_in_text"].tolist(), refs, top_k=3)
            df_missing["suggested_matches"] = df_missing["citation_in_text"].map(sugg)
    else:
        df_c2r, df_r2c = reconcile_numeric(cites, refs)
        df_missing, df_uncited = missing_and_uncited_numeric(df_c2r, refs)

# Summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("In-text citations found", f"{len(cites)}")
col2.metric("Unique in-text citations", f"{len(cite_counter)}")
col3.metric("Reference entries found", f"{len(refs)}")
if style.startswith("APA/Harvard"):
    col4.metric("Uncited references", f"{int((df_r2c['times_cited'] == 0).sum())}" if "times_cited" in df_r2c.columns else "0")
else:
    col4.metric("Uncited references", f"{int((df_r2c['times_cited'] == 0).sum())}" if "times_cited" in df_r2c.columns else "0")

st.divider()

# Missing and uncited panels
left, right = st.columns(2)

with left:
    st.subheader("Cited in-text but missing in References")
    if len(df_missing) == 0:
        st.success("No missing citations detected.")
    else:
        # Ensure we show FULL in-text strings (not just 'Surname, Year')
        st.dataframe(df_missing, width="stretch", hide_index=True)
        st.download_button(
            "Download missing citations (CSV)",
            df_missing.to_csv(index=False).encode("utf-8"),
            file_name="missing_citations.csv",
            mime="text/csv",
        )

with right:
    st.subheader("In References but never cited")
    if len(df_uncited) == 0:
        st.success("No uncited references detected.")
    else:
        st.dataframe(df_uncited, width="stretch", hide_index=True)
        st.download_button(
            "Download uncited references (CSV)",
            df_uncited.to_csv(index=False).encode("utf-8"),
            file_name="uncited_references.csv",
            mime="text/csv",
        )

st.divider()

# Reconciliation tables (requested)
st.subheader("Reconciliation (maps in-text citations to exact references)")

if style.startswith("APA/Harvard"):
    st.markdown("### In-text → Reference")
    st.dataframe(df_c2r, width="stretch", hide_index=True)
    st.download_button(
        "Download in-text → reference (CSV)",
        df_c2r.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_intext_to_reference.csv",
        mime="text/csv",
    )

    st.markdown("### Reference → Cited by")
    st.dataframe(df_r2c, width="stretch", hide_index=True)
    st.download_button(
        "Download reference → in-text (CSV)",
        df_r2c.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_reference_to_intext.csv",
        mime="text/csv",
    )

else:
    st.markdown("### In-text (numbers) → Reference")
    st.dataframe(df_c2r, width="stretch", hide_index=True)
    st.download_button(
        "Download in-text → reference (CSV)",
        df_c2r.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_intext_to_reference.csv",
        mime="text/csv",
    )

    st.markdown("### Reference → Times cited")
    st.dataframe(df_r2c, width="stretch", hide_index=True)
    st.download_button(
        "Download reference → times cited (CSV)",
        df_r2c.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_reference_to_intext.csv",
        mime="text/csv",
    )

st.divider()

# Online verification
if enable_verify and refs:
    st.subheader("Online verification (Crossref + OpenAlex)")
    st.caption("This checks whether a reference looks discoverable online. It does not replace manual verification.")

    to_check = refs[: max_verify]
    ver_rows = []
    with st.spinner("Verifying references online..."):
        for r in to_check:
            v = verify_reference_online(r.raw)
            ver_rows.append(
                {
                    "reference": r.raw,
                    "source": v.get("source", ""),
                    "match_score": v.get("match_score", ""),
                    "title": v.get("title", ""),
                    "year": v.get("year", ""),
                    "doi": v.get("doi", ""),
                    "url": v.get("url", ""),
                }
            )

    df_ver = pd.DataFrame(ver_rows)
    st.dataframe(df_ver, width="stretch", hide_index=True)
    st.download_button(
        "Download verification results (CSV)",
        df_ver.to_csv(index=False).encode("utf-8"),
        file_name="reference_verification.csv",
        mime="text/csv",
    )

st.divider()

# Debug
if show_debug:
    st.subheader("Debug")
    st.markdown("#### Sample parsed citations (first 120)")
    st.dataframe(pd.DataFrame([c.__dict__ for c in cites[:120]]), width="stretch", hide_index=True)

    st.markdown("#### Sample parsed references (first 80)")
    st.dataframe(pd.DataFrame([r.__dict__ for r in refs[:80]]), width="stretch", hide_index=True)

    st.markdown("#### Extracted sections")
    st.text_area("Main text (start)", main_text[:6000], height=200)
    st.text_area("References (start)", ref_text[:6000], height=200)
