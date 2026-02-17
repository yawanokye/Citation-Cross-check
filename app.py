# app.py
# Citation Crosschecker (APA/Harvard + Vancouver)
# - Robust in-text citation capture (keeps full raw strings like "(Alagidede, Tweneboah, & Adam, 2008)")
# - Better reference entry detection from DOCX (References section parsing)
# - Matching that tolerates "&" vs "and", commas around ampersand, and multi-author variants
# - Reconciliation tables: in-text → reference, and reference → cited by

import re
import io
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
from collections import defaultdict, Counter

import streamlit as st
import pandas as pd

# ----------------------------
# Optional PDF support
# ----------------------------
PDF_OK = False
try:
    import pdfplumber  # type: ignore
    PDF_OK = True
except Exception:
    PDF_OK = False

# ----------------------------
# Optional DOCX support
# ----------------------------
DOCX_OK = False
try:
    from docx import Document  # type: ignore
    DOCX_OK = True
except Exception:
    DOCX_OK = False


# ============================
# Data models
# ============================
@dataclass(frozen=True)
class InTextCitation:
    raw: str
    year: str
    surnames: Tuple[str, ...]  # cleaned tokens
    key_primary: str
    key_variants: Tuple[str, ...]  # keys to try (au3->au2->au, etc.)


@dataclass(frozen=True)
class ReferenceEntry:
    raw: str
    year: str
    surnames: Tuple[str, ...]
    key_primary: str
    key_variants: Tuple[str, ...]


# ============================
# Normalisation helpers
# ============================
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})([a-z])?\b")  # allow 1600s in rare cases, plus 2000s + suffix
HEADING_RE = re.compile(r"^\s*(references|reference|bibliography|works cited)\s*$", re.I)

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def norm_punct(s: str) -> str:
    # Keep apostrophes/hyphens in surnames, normalise quotes, remove most other punctuation
    s = (s or "")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = _strip_accents(s)
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[,:;.!?]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def titleish(s: str) -> str:
    s = norm_space(s)
    if not s:
        return s
    return s[0].upper() + s[1:]

def clean_author_token(s: str) -> str:
    # For surnames/org tokens used in keys
    s = norm_punct(s)
    # Drop common fillers
    drop = {"et", "al", "al.", "and"}
    parts = [p for p in s.split() if p and p not in drop]
    if not parts:
        return ""
    # If it's an organisation, keep first 2 words as a stable token
    if len(parts) >= 2 and (parts[0] in {"world", "international", "united"}):
        return (parts[0] + "_" + parts[1]).strip("_")
    # Else keep the last token (typical surname behaviour)
    return parts[-1]

def key_author_year_surnames(surnames: Iterable[str], year: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Builds a primary key and variants.
    - primary = au_<a1>_<year>
    - variants include au2_, au3_ when possible, and always include au_ fallback
    """
    y = norm_space(year)
    ss = [clean_author_token(x) for x in surnames if clean_author_token(x)]
    ss = [x for x in ss if x]
    if not ss:
        primary = f"au_unknown_{y}"
        return primary, (primary,)

    a1 = ss[0]
    variants = []

    # Most specific first
    if len(ss) >= 3:
        variants.append(f"au3_{ss[0]}_{ss[1]}_{ss[2]}_{y}")
    if len(ss) >= 2:
        variants.append(f"au2_{ss[0]}_{ss[1]}_{y}")
    variants.append(f"au_{a1}_{y}")

    primary = f"au_{a1}_{y}"
    # De-dup while keeping order
    seen = set()
    ordered = []
    for k in variants:
        if k not in seen:
            ordered.append(k)
            seen.add(k)
    return primary, tuple(ordered)


# ============================
# DOCX / PDF text extraction
# ============================
def read_docx_text_and_refs(file_bytes: bytes) -> Tuple[str, List[str]]:
    if not DOCX_OK:
        raise RuntimeError("python-docx is not installed. Add 'python-docx' to requirements.txt.")

    doc = Document(io.BytesIO(file_bytes))
    paras = [norm_space(p.text) for p in doc.paragraphs]
    # Remove totally empty
    paras2 = [p for p in paras if p]

    # Detect References heading, then treat subsequent paragraphs as reference entries
    ref_start_idx = None
    for i, p in enumerate(paras2):
        if HEADING_RE.match(p):
            ref_start_idx = i
            break

    if ref_start_idx is None:
        # No explicit References heading, return full text and empty refs
        full_text = "\n".join(paras2)
        return full_text, []

    main_text = "\n".join(paras2[:ref_start_idx])
    ref_block = paras2[ref_start_idx + 1 :]

    # Reference entries are usually one per paragraph in DOCX.
    # Merge continuation lines that look like they belong to previous ref (starts lowercase or digit)
    refs: List[str] = []
    buf = ""
    for p in ref_block:
        if not buf:
            buf = p
            continue
        starts_like_continuation = bool(re.match(r"^\s*(\(|\d+\.|[a-z])", p))
        if starts_like_continuation:
            buf = buf + " " + p
        else:
            refs.append(buf.strip())
            buf = p
    if buf:
        refs.append(buf.strip())

    # Filter out obvious non-refs (short headings)
    refs = [r for r in refs if len(r) >= 15 and not HEADING_RE.match(r)]
    return main_text, refs


def read_pdf_text(file_bytes: bytes) -> str:
    if not PDF_OK:
        raise RuntimeError("pdfplumber is not installed. Add 'pdfplumber' to requirements.txt.")
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return "\n".join(text_parts)


# ============================
# Reference parsing
# ============================
def parse_reference_entry_apa(raw_ref: str) -> Optional[ReferenceEntry]:
    r = norm_space(raw_ref)
    if not r:
        return None

    # Find (YEAR) first
    m = re.search(r"\(\s*((?:1[6-9]\d{2}|20\d{2})(?:[a-z])?)\s*\)", r)
    if not m:
        return None
    year = m.group(1)

    pre = r[: m.start()].strip()

    # Organisation style: no commas, ends with period
    # Person style: "Surname, I., Surname2, I., & Surname3, I."
    surnames: List[str] = []

    if "," in pre:
        # Extract surnames as tokens before commas in author chunks.
        # Split on '&' and 'and' and also handle multiple authors separated by '.,'
        pre2 = pre.replace("&", " and ")
        # Break into author chunks using " and " as separator
        chunks = [c.strip() for c in re.split(r"\s+and\s+", pre2) if c.strip()]
        for ch in chunks:
            # Many refs have "Surname, X. Y."
            # Keep the first token before the first comma as surname.
            parts = [p.strip() for p in ch.split(",") if p.strip()]
            if parts:
                surnames.append(parts[0])
    else:
        # Organisation name: take first ~6 words
        org = re.split(r"\.", pre)[0].strip()
        if org:
            surnames.append(org)

    key_primary, key_variants = key_author_year_surnames(surnames, year)
    return ReferenceEntry(raw=r, year=year, surnames=tuple(surnames), key_primary=key_primary, key_variants=key_variants)


def parse_reference_entries_apa(ref_lines: List[str]) -> List[ReferenceEntry]:
    out = []
    for raw in ref_lines:
        ent = parse_reference_entry_apa(raw)
        if ent:
            out.append(ent)
    return out


# ============================
# In-text citation parsing (APA/Harvard)
# ============================
def _split_multi_cites_inside_parentheses(s: str) -> List[str]:
    # Separate multiple citations in one parenthetical: "(A, 2010; B & C, 2012)"
    # Keep semicolon separation primarily
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return parts if parts else [s.strip()]

def _extract_year(text: str) -> Optional[str]:
    m = YEAR_RE.search(text)
    if not m:
        return None
    return (m.group(1) + (m.group(2) or "")).strip()

def _extract_authors_from_cite_chunk(chunk: str) -> List[str]:
    """
    chunk examples:
      "Alagidede, Tweneboah, & Adam, 2008"
      "Kofi et al., 2020"
      "World Health Organization, 2019"
      "Mundell, 1961"
    """
    c = norm_space(chunk)
    # Remove leading/trailing parentheses
    c = c.strip("()[]{} ")
    # Remove year and any locator after year
    # keep only part before the year
    y = _extract_year(c)
    if not y:
        return []
    pre = c.split(y)[0].strip()
    # remove trailing punctuation/comma
    pre = pre.rstrip(",").strip()

    # Replace ampersand to unify
    pre = pre.replace("&", " and ")
    # Remove "see" and similar leading words
    pre = re.sub(r"^(see|e\.g\.|for example)\s+", "", pre, flags=re.I).strip()

    # If "et al." is used, keep only first author token before it
    if re.search(r"\bet\s+al\.?\b", pre, flags=re.I):
        first = re.split(r"\bet\s+al\.?\b", pre, flags=re.I)[0].strip().rstrip(",")
        # first might be "Kofi" or "Kofi," etc.
        if first:
            # If it is "Surname," keep just before comma
            if "," in first:
                first = first.split(",")[0].strip()
            return [first]

    # Split authors by commas and "and"
    # For APA in-text, multiple authors often written "A, B, & C" where commas separate surnames
    pieces = [p.strip() for p in re.split(r"\s+and\s+|,", pre) if p.strip()]

    # Filter out initials-only fragments
    cleaned = []
    for p in pieces:
        # drop single-letter initials
        if re.fullmatch(r"[A-Z]\.?", p):
            continue
        cleaned.append(p)
    return cleaned


def parse_in_text_citations_apa(full_text: str) -> List[InTextCitation]:
    text = full_text

    cites: List[InTextCitation] = []

    # 1) Parenthetical citations: (...) containing a year
    # Grab parenthetical groups that contain a year-like pattern
    for m in re.finditer(r"\(([^()]{0,300}?\b(?:1[6-9]\d{2}|20\d{2})[a-z]?\b[^()]*)\)", text):
        inside = m.group(1)
        for chunk in _split_multi_cites_inside_parentheses(inside):
            y = _extract_year(chunk)
            if not y:
                continue
            authors = _extract_authors_from_cite_chunk(chunk)
            key_primary, key_variants = key_author_year_surnames(authors, y)
            raw_full = "(" + norm_space(chunk) + ")"
            cites.append(
                InTextCitation(
                    raw=raw_full,
                    year=y,
                    surnames=tuple(authors),
                    key_primary=key_primary,
                    key_variants=key_variants,
                )
            )

    # 2) Narrative citations: Author ... (YEAR)
    # Examples:
    #   Alagidede, Tweneboah, & Adam (2008)
    #   Mundell (1961)
    #   Zehirun et al. (2016)
    # Capture up to 120 chars before (YEAR), but stop at sentence breaks/newlines
    for m in re.finditer(r"(?<!\()([^\n\.]{1,120}?)\(\s*((?:1[6-9]\d{2}|20\d{2})(?:[a-z])?)\s*\)", text):
        left = norm_space(m.group(1))
        y = m.group(2)
        # Avoid catching things like "Figure 2 (2019)" by requiring at least one letter and not starting with "figure/table/section"
        if re.match(r"^(figure|table|section|appendix)\b", left, flags=re.I):
            continue
        if not re.search(r"[A-Za-z]", left):
            continue

        # Take only the author-ish tail: last ~80 chars
        authorish = left[-80:].strip()
        # Remove leading connectors like "by", "in", "see", "as"
        authorish = re.sub(r"^(by|in|see|as)\s+", "", authorish, flags=re.I).strip()
        # Remove trailing possessives/verbs fragments
        authorish = re.sub(r"\b(argues|shows|finds|notes|suggests|proposes)\b.*$", "", authorish, flags=re.I).strip()

        authors = _extract_authors_from_cite_chunk(authorish + ", " + y)
        if not authors:
            continue
        key_primary, key_variants = key_author_year_surnames(authors, y)
        raw_full = f"{authorish} ({y})"
        cites.append(
            InTextCitation(
                raw=norm_space(raw_full),
                year=y,
                surnames=tuple(authors),
                key_primary=key_primary,
                key_variants=key_variants,
            )
        )

    # De-duplicate by (raw, year) while preserving counts later via Counter on key_primary+raw
    uniq = []
    seen = set()
    for c in cites:
        k = (c.raw, c.year)
        if k not in seen:
            uniq.append(c)
            seen.add(k)
    return uniq


# ============================
# Vancouver parsing (basic)
# ============================
def parse_in_text_citations_vancouver(text: str) -> List[str]:
    # Capture [1], [1-3], [1,2,5]
    hits = re.findall(r"\[(\s*\d+(?:\s*[-,]\s*\d+)*\s*)\]", text)
    out = []
    for h in hits:
        out.append("[" + re.sub(r"\s+", "", h) + "]")
    return out

def parse_reference_entries_vancouver(ref_lines: List[str]) -> List[str]:
    # Very simple: keep lines that start with a number or look like a ref
    out = []
    for r in ref_lines:
        rr = norm_space(r)
        if re.match(r"^\d+\.", rr) or re.match(r"^\d+\s", rr):
            out.append(rr)
        elif YEAR_RE.search(rr) and len(rr) > 25:
            out.append(rr)
    return out


# ============================
# Matching and reporting (APA/Harvard)
# ============================
def build_reference_index(refs: List[ReferenceEntry]) -> Dict[str, List[ReferenceEntry]]:
    ref_by_key: Dict[str, List[ReferenceEntry]] = defaultdict(list)
    for r in refs:
        for k in r.key_variants:
            ref_by_key[k].append(r)
    return ref_by_key

def best_match_for_citation(c: InTextCitation, ref_by_key: Dict[str, List[ReferenceEntry]]) -> List[ReferenceEntry]:
    # Try variants from most specific to fallback
    for k in c.key_variants:
        hits = ref_by_key.get(k, [])
        if hits:
            return hits
    # Special fallback: if au2_* try au_* (already included), if au3_* try au2_ then au_ (already included)
    return []

def missing_and_uncited_apa(
    cites: List[InTextCitation], refs: List[ReferenceEntry]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_key = build_reference_index(refs)

    # Count citations by their RAW string (users care about exact in-text strings) and also keep match suggestions
    cite_counter = Counter([c.raw for c in cites])

    rows_missing = []
    for raw_cite, cnt in cite_counter.items():
        # Find a representative citation object for this raw cite
        c_obj = next((c for c in cites if c.raw == raw_cite), None)
        if not c_obj:
            continue
        hits = best_match_for_citation(c_obj, ref_by_key)
        if not hits:
            rows_missing.append(
                {
                    "citation_in_text": raw_cite,
                    "count_in_text": cnt,
                    "suggested_matches": "",
                }
            )

    df_missing = pd.DataFrame(rows_missing).sort_values(["count_in_text", "citation_in_text"], ascending=[False, True]) if rows_missing else pd.DataFrame(
        columns=["citation_in_text", "count_in_text", "suggested_matches"]
    )

    # Determine which references are cited (by key hits)
    cited_ref_ids = set()
    for c in cites:
        hits = best_match_for_citation(c, ref_by_key)
        for h in hits:
            cited_ref_ids.add(id(h))

    rows_uncited = []
    for r in refs:
        if id(r) not in cited_ref_ids:
            rows_uncited.append({"reference_full": r.raw})

    df_uncited = pd.DataFrame(rows_uncited) if rows_uncited else pd.DataFrame(columns=["reference_full"])
    return df_missing, df_uncited


# ============================
# Reconciliation tables (APA/Harvard)
# ============================
def reconcile_author_year(
    cites: List[InTextCitation], refs: List[ReferenceEntry]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - df_cite_to_ref: each in-text cite mapped to best reference (or ambiguous / not found)
    - df_ref_to_cites: each reference mapped to citations that point to it
    """
    ref_by_key = build_reference_index(refs)

    rows = []
    for c in cites:
        hits = best_match_for_citation(c, ref_by_key)

        if not hits:
            rows.append({"in_text": c.raw, "status": "not_found", "matched_reference": ""})
        elif len(hits) == 1:
            rows.append({"in_text": c.raw, "status": "matched", "matched_reference": hits[0].raw})
        else:
            rows.append(
                {
                    "in_text": c.raw,
                    "status": f"ambiguous ({len(hits)})",
                    "matched_reference": " || ".join(h.raw[:220] for h in hits),
                }
            )

    df_cite_to_ref = pd.DataFrame(rows)

    # Reverse mapping: reference -> cited by (raw cites)
    cite_group = defaultdict(list)
    for c in cites:
        hits = best_match_for_citation(c, ref_by_key)
        # Attach this raw cite to every hit it maps to
        for h in hits:
            cite_group[id(h)].append(c.raw)

    ref_rows = []
    for r in refs:
        cited_by = cite_group.get(id(r), [])
        ref_rows.append(
            {
                "reference": r.raw,
                "times_cited": len(cited_by),
                "cited_by": "; ".join(cited_by[:12]) + (" ..." if len(cited_by) > 12 else ""),
            }
        )
    df_ref_to_cites = pd.DataFrame(ref_rows).sort_values(["times_cited"], ascending=False)

    return df_cite_to_ref, df_ref_to_cites


# ============================
# UI
# ============================
st.set_page_config(page_title="Citation Crosschecker", layout="wide")
st.title("Citation Crosschecker")
st.caption("Checks in-text citations against reference entries. Best results with APA/Harvard and Vancouver styles.")

with st.sidebar:
    st.header("Settings")
    style = st.selectbox(
        "Citation style",
        ["APA/Harvard (author–year)", "Vancouver (numeric)"],
        index=0,
    )
    st.write("")
    st.subheader("Upload")
    up = st.file_uploader("Upload a manuscript (DOCX or PDF)", type=["docx", "pdf"])

if not up:
    st.info("Upload a DOCX (recommended) or PDF to begin.")
    st.stop()

file_bytes = up.read()
file_name = up.name.lower()

# Extract text + refs
main_text = ""
ref_lines: List[str] = []

try:
    if file_name.endswith(".docx"):
        main_text, ref_lines = read_docx_text_and_refs(file_bytes)
    elif file_name.endswith(".pdf"):
        # PDF: we can extract text, but references section splitting is harder.
        # We'll still attempt: look for a "References" heading inside text.
        full = read_pdf_text(file_bytes)
        # crude split on References heading
        chunks = re.split(r"\n\s*(references|bibliography|works cited)\s*\n", full, flags=re.I)
        if len(chunks) >= 3:
            main_text = chunks[0]
            ref_block = "\n".join(chunks[2:])
            # split references by line breaks, then merge lightly
            raw_lines = [norm_space(x) for x in ref_block.split("\n") if norm_space(x)]
            # merge continuation lines
            merged = []
            buf = ""
            for ln in raw_lines:
                if not buf:
                    buf = ln
                    continue
                if re.match(r"^\(?\d+\)?[.\)]?\s", ln) or YEAR_RE.search(ln):
                    merged.append(buf)
                    buf = ln
                else:
                    buf = buf + " " + ln
            if buf:
                merged.append(buf)
            ref_lines = merged
        else:
            main_text = full
            ref_lines = []
    else:
        st.error("Unsupported file type.")
        st.stop()
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if style.startswith("APA/Harvard"):
    cites = parse_in_text_citations_apa(main_text)
    refs = parse_reference_entries_apa(ref_lines)

    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("In-text citations found", len(cites))
    col2.metric("Reference entries found", len(refs))
    col3.metric("References heading detected", "Yes" if ref_lines else "No / uncertain")

    if not ref_lines:
        st.warning(
            "I could not confidently locate a References section. In DOCX, add a heading like 'References'. "
            "In PDF, extraction is more fragile."
        )

    # Missing / Uncited
    st.divider()
    st.subheader("Cross-check results")

    df_missing, df_uncited = missing_and_uncited_apa(cites, refs)

    left, right = st.columns(2)

    with left:
        st.markdown("### Cited in-text but missing in References")
        st.dataframe(df_missing, use_container_width=True, height=420)
        st.download_button(
            "Download missing (CSV)",
            df_missing.to_csv(index=False).encode("utf-8"),
            file_name="missing_in_references.csv",
            mime="text/csv",
        )

    with right:
        st.markdown("### In References but never cited")
        st.dataframe(df_uncited, use_container_width=True, height=420)
        st.download_button(
            "Download uncited (CSV)",
            df_uncited.to_csv(index=False).encode("utf-8"),
            file_name="uncited_references.csv",
            mime="text/csv",
        )

    # Reconciliation tables
    st.divider()
    st.subheader("Reconciliation (which citation maps to which reference)")

    df_c2r, df_r2c = reconcile_author_year(cites, refs)

    st.markdown("### In-text → Reference")
    st.dataframe(df_c2r, use_container_width=True, height=420)
    st.download_button(
        "Download in-text to reference mapping (CSV)",
        df_c2r.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_intext_to_reference.csv",
        mime="text/csv",
    )

    st.markdown("### Reference → Cited by")
    st.dataframe(df_r2c, use_container_width=True, height=420)
    st.download_button(
        "Download reference to in-text mapping (CSV)",
        df_r2c.to_csv(index=False).encode("utf-8"),
        file_name="reconciliation_reference_to_intext.csv",
        mime="text/csv",
    )

    # Diagnostics panel
    with st.expander("Diagnostics (keys and parsing preview)"):
        st.write("Sample parsed citations (first 20):")
        diag_c = []
        for c in cites[:20]:
            diag_c.append(
                {
                    "raw": c.raw,
                    "year": c.year,
                    "surnames": ", ".join(c.surnames),
                    "primary_key": c.key_primary,
                    "variants": " | ".join(c.key_variants),
                }
            )
        st.dataframe(pd.DataFrame(diag_c), use_container_width=True)

        st.write("Sample parsed references (first 20):")
        diag_r = []
        for r in refs[:20]:
            diag_r.append(
                {
                    "raw": r.raw[:260],
                    "year": r.year,
                    "surnames": ", ".join(r.surnames),
                    "primary_key": r.key_primary,
                    "variants": " | ".join(r.key_variants),
                }
            )
        st.dataframe(pd.DataFrame(diag_r), use_container_width=True)

else:
    # Vancouver basic mode
    cites_v = parse_in_text_citations_vancouver(main_text)
    refs_v = parse_reference_entries_vancouver(ref_lines)

    col1, col2 = st.columns(2)
    col1.metric("In-text bracket citations found", len(cites_v))
    col2.metric("Reference-like entries found", len(refs_v))

    st.divider()
    st.subheader("What this mode does now")
    st.info(
        "Vancouver support here is basic. It detects bracket citations like [1], [1-3], [2,5]. "
        "Accurate mapping to a numbered reference list needs stricter parsing of the reference numbering."
    )

    st.markdown("### Sample in-text citations")
    st.dataframe(pd.DataFrame({"citation_in_text": cites_v[:80]}), use_container_width=True)

    st.markdown("### Sample reference entries")
    st.dataframe(pd.DataFrame({"reference_full": refs_v[:80]}), use_container_width=True)
