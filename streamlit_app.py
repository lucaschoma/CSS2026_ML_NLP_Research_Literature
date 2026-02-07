import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import feedparser
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from dateutil import parser as dateparser


# Full-text extraction (works well across many news/blog sites)
import trafilatura

# Lightweight extractive summarization (fast, local)
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

# NLTK punkt is needed by sumy tokenizer
import nltk
try:
    nltk.data.find("tokenizers/punkt")
    nltk.download('punkt_tab')
except LookupError:
    nltk.download("punkt", quiet=True)


# -----------------------------
# Config: Sources (RSS preferred)
# -----------------------------
DEFAULT_SOURCES = [
    # Research & preprints
    {"name": "arXiv CS.CL (NLP)", "type": "rss", "url": "http://export.arxiv.org/rss/cs.CL"},
    {"name": "arXiv CS.LG (ML)",  "type": "rss", "url": "http://export.arxiv.org/rss/cs.LG"},

    # Industry labs & blogs (many have RSS)
    {"name": "Google AI Blog", "type": "rss", "url": "https://ai.googleblog.com/feeds/posts/default"},
    {"name": "OpenAI Blog",    "type": "rss", "url": "https://openai.com/blog/rss.xml"},
    {"name": "Hugging Face Blog", "type": "rss", "url": "https://huggingface.co/blog/feed.xml"},
    {"name": "DeepMind Blog",  "type": "rss", "url": "https://deepmind.google/discover/blog/rss.xml"},
    {"name": "Microsoft Research Blog", "type": "rss", "url": "https://www.microsoft.com/en-us/research/feed/"},

    # Community / engineering
    {"name": "Towards Data Science", "type": "rss", "url": "https://towardsdatascience.com/feed"},
    {"name": "The Batch (DeepLearning.AI)", "type": "rss", "url": "https://www.deeplearning.ai/the-batch/feed/"},
]


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Article:
    source: str
    title: str
    link: str
    published: Optional[datetime]
    author: str = ""
    tags: str = ""
    summary: str = ""          # RSS/HTML summary
    fulltext: str = ""         # Extracted full text
    auto_summary: str = ""     # Summarized content


# -----------------------------
# Helpers
# -----------------------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = BeautifulSoup(s, "html.parser").get_text(" ", strip=True)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_parse_date(entry: dict) -> Optional[datetime]:
    """Try parsing various date fields from RSS entries."""
    for key in ("published", "updated", "created", "date"):
        if key in entry and entry[key]:
            try:
                dt = dateparser.parse(entry[key])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue

    # Some feeds provide structured time
    for key in ("published_parsed", "updated_parsed"):
        if key in entry and entry[key]:
            try:
                dt = datetime(*entry[key][:6], tzinfo=timezone.utc)
                return dt
            except Exception:
                continue

    return None



def extract_tags(entry: dict) -> str:
    """
    Robust tag extraction supporting:
    - list[str]                 (our sanitized format)
    - list[dict] with 'term'    (raw feedparser format)
    - missing/None
    """
    tags = entry.get("tags", [])
    if not tags:
        return ""

    out = []
    for t in tags:
        if isinstance(t, dict):
            term = t.get("term", "")
            if term:
                out.append(term)
        elif isinstance(t, str):
            if t.strip():
                out.append(t.strip())

    # de-duplicate while preserving order
    seen = set()
    unique = []
    for t in out:
        if t not in seen:
            unique.append(t)
            seen.add(t)

    return ", ".join(unique)


from typing import Any

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_feed_entries(url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Fetch RSS/Atom and return ONLY pickle-safe entry dicts.
    We avoid returning feedparser.FeedParserDict because it may not be pickleable.
    """
    parsed = feedparser.parse(url)
    entries = parsed.get("entries", [])[:limit]

    safe_entries: List[Dict[str, Any]] = []
    for e in entries:
        safe_entries.append({
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "author": e.get("author", ""),
            "summary": e.get("summary", "") or e.get("description", ""),
            # keep both string and parsed tuples for robust date parsing
            "published": e.get("published", "") or e.get("updated", ""),
            "updated": e.get("updated", ""),
            "published_parsed": tuple(e.get("published_parsed") or ()),
            "updated_parsed": tuple(e.get("updated_parsed") or ()),
            # normalize tags into a simple list[str]
            "tags": [t.get("term", "") for t in (e.get("tags", []) or []) if isinstance(t, dict) and t.get("term")],
        })

    return safe_entries


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fulltext(url: str) -> str:
    """
    Download and extract main content with trafilatura.
    Falls back to empty string if extraction fails.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_recall=True
        )
        return extracted or ""
    except Exception:
        return ""


def lexrank_summarize(text: str, sentences: int = 4) -> str:
    """Fast extractive summary using LexRank."""
    text = clean_text(text)
    if not text:
        return ""
    # Guard for very short text
    if len(text.split()) < 80:
        return textwrap.shorten(text, width=400, placeholder="…")

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences)
    return " ".join(str(s) for s in summary_sentences).strip()


def try_transformers_summarize(
    text: str,
    max_new_tokens: int = 160,
    min_new_tokens: int = 60
) -> Optional[str]:
    """
    Abstractive summary using HF transformers WITHOUT pipeline task strings.
    This avoids KeyError when 'summarization' isn't registered in your transformers build.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception:
        return None

    text = clean_text(text)
    if len(text) < 200:
        return textwrap.shorten(text, width=400, placeholder="…")

    model_name = "sshleifer/distilbart-cnn-12-6"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Optional: move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        chunks = chunk_text(text, max_chars=2500)
        summaries = []

        for ch in chunks[:3]:
            inputs = tokenizer(
                ch,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    num_beams=4,
                    do_sample=False,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    length_penalty=1.0,
                    early_stopping=True
                )

            summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        return " ".join(summaries).strip()

    except Exception:
        # If model can't be downloaded/loaded in your environment, return None and fall back to LexRank.
        return None


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """Split text into roughly max_chars chunks at sentence boundaries."""
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, buff = [], []
    size = 0
    for s in sentences:
        if size + len(s) + 1 > max_chars and buff:
            chunks.append(" ".join(buff))
            buff, size = [], 0
        buff.append(s)
        size += len(s) + 1
    if buff:
        chunks.append(" ".join(buff))
    return chunks


def within_days(dt: Optional[datetime], days: int) -> bool:
    if dt is None:
        return True  # If unknown date, keep unless user wants strict filtering
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return dt >= cutoff


def articles_to_df(articles: List[Article]) -> pd.DataFrame:
    rows = []
    for a in articles:
        rows.append({
            "source": a.source,
            "title": a.title,
            "link": a.link,
            "published": a.published.isoformat() if a.published else "",
            "author": a.author,
            "tags": a.tags,
            "rss_summary": a.summary,
            "auto_summary": a.auto_summary,
        })
    return pd.DataFrame(rows)


# -----------------------------
# App Logic
# -----------------------------
def load_articles(
    sources: List[Dict],
    max_items_per_source: int,
    only_last_days: int,
    keyword: str,
    use_fulltext: bool,
) -> List[Article]:
    articles: List[Article] = []

    for src in sources:
        if src.get("type") != "rss":
            continue

        entries = fetch_feed_entries(src["url"], limit=max_items_per_source)

        for entry in entries:
            title = clean_text(entry.get("title", "Untitled"))
            link = entry.get("link", "")
            published = safe_parse_date(entry)
            author = clean_text(entry.get("author", "")) if entry.get("author") else ""
            tags = extract_tags(entry)
            summary = clean_text(entry.get("summary", "") or entry.get("description", ""))

            # Time filter
            if not within_days(published, only_last_days):
                continue

            # Keyword filter on title + summary
            haystack = f"{title} {summary}".lower()
            if keyword and keyword.lower() not in haystack:
                continue

            art = Article(
                source=src["name"],
                title=title,
                link=link,
                published=published,
                author=author,
                tags=tags,
                summary=summary
            )

            # Fetch full text if enabled
            if use_fulltext and link:
                art.fulltext = fetch_fulltext(link)

            articles.append(art)

    # Sort by date desc (unknown dates at bottom)
    articles.sort(key=lambda a: a.published or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
    return articles


def summarize_articles(
    articles: List[Article],
    prefer_transformers: bool,
    summary_sentences: int
) -> List[Article]:
    for a in articles:
        base = a.fulltext if a.fulltext else a.summary
        if not base:
            a.auto_summary = ""
            continue

        if prefer_transformers:
            t = try_transformers_summarize(base)
            if t:
                a.auto_summary = t
                continue

        a.auto_summary = lexrank_summarize(base, sentences=summary_sentences)

    return articles


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ML/NLP Research Portfolio", layout="wide")

st.title("CSS2026 Final Streamlit App: Machine Learning (ML) and Natural Language Processing (NLP) Research Portfolio")

st.caption("Aggregate research articles/blog posts, auto-summarize, save to a portfolio, and export.")

st.write(
    "This is an a collection of online research articles and their summary for Machine Learning (ML) and Natural Language Processing (NLP). " \
    "I use RSS to find and aggregate content from various sources, then apply extractive and abstractive summarization techniques to distill the key insights. " \
    "You can save interesting articles to this portfolio, add notes/tags, and export for later reference."
)

# Session state for portfolio
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []  # list of dicts {title, link, source, published, notes, tags}

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    max_items_per_source = st.slider("Max items per source", 2, 20, 5, 2)
    only_last_days = st.slider("Show items from last (days)", 5, 365, 90, 5)
    keyword = st.text_input("Keyword filter (title/summary)", "")
    use_fulltext = st.checkbox("Try full-text extraction (recommended)", True)

    st.subheader("Summarization")
    prefer_transformers = st.checkbox("Use Transformers (abstractive, heavier)", False)
    summary_sentences = st.slider("Extractive summary length (sentences)", 2, 10, 4, 2)

    st.subheader("Sources")
    if st.checkbox("Show/Edit sources"):
        st.info("Tip: Add RSS/Atom feed URLs. RSS is the most reliable method.")
        sources_df = pd.DataFrame(DEFAULT_SOURCES)
        edited = st.data_editor(sources_df, num_rows="dynamic", width="stretch")
        sources = edited.to_dict(orient="records")
    else:
        sources = DEFAULT_SOURCES

    st.divider()
    page = st.radio("Navigate", ["Feed", "Portfolio", "About"], index=0)


if page == "Feed":
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        with st.spinner("Fetching articles…"):
            articles = load_articles(
                sources=sources,
                max_items_per_source=max_items_per_source,
                only_last_days=only_last_days,
                keyword=keyword,
                use_fulltext=use_fulltext,
            )

        st.success(f"Loaded {len(articles)} items")

        with st.spinner("Summarizing…"):
            articles = summarize_articles(
                articles=articles,
                prefer_transformers=prefer_transformers,
                summary_sentences=summary_sentences
            )

        # Display articles as cards
        for i, a in enumerate(articles):
            published_str = a.published.astimezone(timezone.utc).strftime("%Y-%m-%d") if a.published else "Unknown date"
            with st.container(border=True):
                top = st.columns([5, 2, 2])
                with top[0]:
                    st.markdown(f"### {a.title}")
                    st.caption(f"**{a.source}** • {published_str}" + (f" • {a.author}" if a.author else ""))
                with top[1]:
                    if a.tags:
                        st.write("**Tags:**")
                        st.write(a.tags)
                with top[2]:
                    if a.link:
                        st.link_button("Open article ↗", a.link, width='stretch')

                if a.auto_summary:
                    st.write(a.auto_summary)
                else:
                    st.write("_No summary available._")

                with st.expander("Show details"):
                    if a.summary:
                        st.write("**RSS summary:**")
                        st.write(a.summary)
                    if a.fulltext:
                        st.write("**Extracted full text (snippet):**")
                        st.write(textwrap.shorten(clean_text(a.fulltext), width=1200, placeholder="…"))

                # Save to portfolio
                save_cols = st.columns([1, 4])
                with save_cols[0]:
                    if st.button("Save", key=f"save_{i}", width='stretch'):
                        st.session_state.portfolio.append({
                            "title": a.title,
                            "link": a.link,
                            "source": a.source,
                            "published": a.published.isoformat() if a.published else "",
                            "tags": a.tags,
                            "notes": ""
                        })
                        st.toast("Saved to portfolio", icon="*<3")
                with save_cols[1]:
                    st.caption("Save items you want to track, annotate, and export.")

    with colB:
        st.subheader("Quick stats")
        if "articles" in locals() and articles:
            df = articles_to_df(articles)
            st.dataframe(df[["published", "source", "title"]], width='stretch', height=420)

            # Counts by source
            counts = df["source"].value_counts().reset_index()
            counts.columns = ["source", "count"]
            st.bar_chart(counts.set_index("source"))

        else:
            st.info("No articles found. Try increasing the time range or removing the keyword filter.")


elif page == "Portfolio":
    st.header("Your Saved Portfolio")

    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Go to **Feed** and click \"*<3\" Save on items you like.")
    else:
        pf = pd.DataFrame(st.session_state.portfolio)
        st.caption(f"{len(pf)} saved items")

        # Edit notes inline
        st.subheader("Edit notes")
        edited_pf = st.data_editor(pf, width='stretch', num_rows="dynamic")
        st.session_state.portfolio = edited_pf.to_dict(orient="records")

        st.divider()

        st.subheader("Export")
        csv = edited_pf.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="ml_nlp_portfolio.csv", mime="text/csv")

        # Simple analytics
        st.subheader("Portfolio insights")
        left, right = st.columns(2)
        with left:
            by_source = edited_pf["source"].value_counts()
            st.write("**Saved by source**")
            st.bar_chart(by_source)
        with right:
            # Tags are comma-separated; explode them
            tags_series = edited_pf["tags"].fillna("").astype(str)
            all_tags = []
            for t in tags_series:
                all_tags.extend([x.strip() for x in t.split(",") if x.strip()])
            if all_tags:
                tags_counts = pd.Series(all_tags).value_counts().head(15)
                st.write("**Top tags**")
                st.bar_chart(tags_counts)
            else:
                st.info("No tags found to chart.")


else:  # About
    st.header("About")
    st.markdown(
        """
**What this app does**
- Pulls articles from **RSS/Atom feeds** (best practice for structured content).
- Optionally extracts article full text using **trafilatura**.
- Summarizes using **LexRank** (fast extractive) or optional **Transformers** (abstractive).

**Notes & Responsible Use**
- Always respect website Terms of Service and robots.txt.
- RSS feeds are the most website-friendly method of aggregation.
- Full-text extraction may not work on paywalled/JS-heavy pages.

**Customize**
- Use the sidebar **Show/Edit sources** to add your own feeds (lab blogs, company research blogs, newsletters, etc.)
        """
    )
