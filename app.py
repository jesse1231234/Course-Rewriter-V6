import os
import re
import html as html_lib
import textwrap
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from bs4 import BeautifulSoup, Tag


# =============================================================================
# CONFIG / CLIENTS
# =============================================================================

def get_ai_client() -> OpenAI:
    base_url = st.secrets.get("OPENAI_BASE_URL", None) or os.getenv("OPENAI_BASE_URL")
    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    if not base_url or not api_key:
        st.error(
            "OpenAI configuration missing. "
            "Please set OPENAI_BASE_URL and OPENAI_API_KEY in Streamlit secrets or environment."
        )
        st.stop()
    return OpenAI(base_url=base_url, api_key=api_key)


def get_model_name() -> str:
    model_name = st.secrets.get("OPENAI_MODEL", None) or os.getenv("OPENAI_MODEL")
    if not model_name:
        st.error(
            "OPENAI_MODEL is not set. "
            "Set it to your model deployment name from Azure (Deployment Info → Name)."
        )
        st.stop()
    return model_name


def get_canvas_config() -> tuple[str, str]:
    base_url = st.secrets.get("CANVAS_BASE_URL", None) or os.getenv("CANVAS_BASE_URL")
    token = st.secrets.get("CANVAS_API_TOKEN", None) or os.getenv("CANVAS_API_TOKEN")
    if not base_url or not token:
        st.error(
            "Canvas API configuration missing. Please set CANVAS_BASE_URL and "
            "CANVAS_API_TOKEN in Streamlit secrets or environment."
        )
        st.stop()
    return base_url.rstrip("/"), token


# =============================================================================
# CANVAS HELPERS
# =============================================================================

def canvas_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _get_next_link(resp: requests.Response) -> Optional[str]:
    link = resp.headers.get("Link", "")
    for part in link.split(","):
        if 'rel="next"' in part:
            return part[part.find("<") + 1 : part.find(">")]
    return None


def get_course(base_url: str, token: str, course_id: str) -> Dict[str, Any]:
    url = f"{base_url}/api/v1/courses/{course_id}"
    resp = requests.get(url, headers=canvas_headers(token))
    resp.raise_for_status()
    return resp.json()


def get_pages(base_url: str, token: str, course_id: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    headers = canvas_headers(token)
    items: List[Dict[str, Any]] = []
    url = f"{base_url}/api/v1/courses/{course_id}/pages"
    params = {"per_page": 100}

    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()

        for page in resp.json():
            detail_url = f"{base_url}/api/v1/courses/{course_id}/pages/{page['url']}"
            detail_resp = requests.get(detail_url, headers=headers)
            detail_resp.raise_for_status()
            items.append(detail_resp.json())

            if max_items and len(items) >= max_items:
                return items

        url = _get_next_link(resp)
        params = None

    return items


def get_assignments(base_url: str, token: str, course_id: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    headers = canvas_headers(token)
    items: List[Dict[str, Any]] = []
    url = f"{base_url}/api/v1/courses/{course_id}/assignments"
    params = {"per_page": 100}

    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        items.extend(resp.json())

        if max_items and len(items) >= max_items:
            return items[:max_items]

        url = _get_next_link(resp)
        params = None

    return items


def get_discussions(base_url: str, token: str, course_id: str, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    headers = canvas_headers(token)
    items: List[Dict[str, Any]] = []
    url = f"{base_url}/api/v1/courses/{course_id}/discussion_topics"
    params = {"per_page": 100}

    while url:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        items.extend(resp.json())

        if max_items and len(items) >= max_items:
            return items[:max_items]

        url = _get_next_link(resp)
        params = None

    return items


def update_page_html(base_url: str, token: str, course_id: str, url_slug: str, html: str) -> None:
    endpoint = f"{base_url}/api/v1/courses/{course_id}/pages/{url_slug}"
    payload = {"wiki_page": {"body": html}}
    resp = requests.put(endpoint, headers=canvas_headers(token), json=payload)
    resp.raise_for_status()


def update_assignment_html(base_url: str, token: str, course_id: str, assignment_id: int, html: str) -> None:
    endpoint = f"{base_url}/api/v1/courses/{course_id}/assignments/{assignment_id}"
    payload = {"assignment": {"description": html}}
    resp = requests.put(endpoint, headers=canvas_headers(token), json=payload)
    resp.raise_for_status()


def update_discussion_html(base_url: str, token: str, course_id: str, topic_id: int, html: str) -> None:
    endpoint = f"{base_url}/api/v1/courses/{course_id}/discussion_topics/{topic_id}"
    payload = {"message": html}
    resp = requests.put(endpoint, headers=canvas_headers(token), json=payload)
    resp.raise_for_status()


# =============================================================================
# DESIGNPLUS: Deterministic iframe -> accordion enforcement
# =============================================================================

DP_ACCORDION_WRAPPER_CLASS = "dp-panels-wrapper dp-accordion-default"
DP_PANEL_GROUP_CLASS = "dp-panel-group"
DP_PANEL_HEADING_CLASS = "dp-panel-heading"
DP_PANEL_CONTENT_CLASS = "dp-panel-content"
DP_EMBED_WRAPPER_CLASS = "dp-embed-wrapper"


def _is_inside_dp_accordion(tag: Tag) -> bool:
    """
    True if tag is inside a dp-panels-wrapper container.
    """
    p = tag.parent
    while p is not None:
        if isinstance(p, Tag):
            cls = " ".join(p.get("class", [])).strip()
            if "dp-panels-wrapper" in cls.split():
                return True
            if "dp-panels-wrapper" in cls:  # covers multi-class string join cases
                return True
        p = p.parent
    return False


def _make_dp_accordion_from_iframes(soup: BeautifulSoup, iframes: List[Tag], title_prefix: str = "Panel") -> Tag:
    """
    Build a dp-panels-wrapper accordion where each iframe becomes its own panel.
    Uses the canonical structure you provided.
    """
    wrapper = soup.new_tag("div")
    wrapper["class"] = DP_ACCORDION_WRAPPER_CLASS.split()

    for idx, iframe in enumerate(iframes, start=1):
        panel_group = soup.new_tag("div")
        panel_group["class"] = [DP_PANEL_GROUP_CLASS]

        heading = soup.new_tag("h3")
        heading["class"] = [DP_PANEL_HEADING_CLASS]
        heading.string = f"{title_prefix} {idx}"

        content = soup.new_tag("div")
        content["class"] = [DP_PANEL_CONTENT_CLASS]

        embed = soup.new_tag("div")
        embed["class"] = [DP_EMBED_WRAPPER_CLASS]

        # Move iframe into embed wrapper (preserve attributes)
        iframe.extract()
        embed.append(iframe)

        content.append(embed)
        panel_group.append(heading)
        panel_group.append(content)
        wrapper.append(panel_group)

    return wrapper


def enforce_iframes_in_dp_accordions(html: str) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministically wrap any iframe(s) not already in a dp accordion into a canonical dp accordion.
    If multiple orphan iframes exist in one container, they will be grouped into a single accordion.
    """
    html = (html or "").strip()
    if not html:
        return html, {"wrapped_iframes": 0, "grouped_accordions_created": 0}

    soup = BeautifulSoup(html, "html.parser")
    root = soup.body if soup.body else soup

    # Find orphan iframes (not already in dp accordion)
    orphan_iframes = []
    for iframe in root.find_all("iframe"):
        if not _is_inside_dp_accordion(iframe):
            orphan_iframes.append(iframe)

    if not orphan_iframes:
        return str(root), {"wrapped_iframes": 0, "grouped_accordions_created": 0}

    # Strategy: group orphans by their nearest reasonable container (parent block),
    # so we don't pull iframes from entirely unrelated places into one accordion.
    grouped: Dict[int, List[Tag]] = {}
    for iframe in orphan_iframes:
        parent = iframe.parent
        # choose a stable ancestor that is a block-ish element
        while parent and isinstance(parent, Tag) and parent.name in {"span", "strong", "em", "b", "i"}:
            parent = parent.parent
        key = id(parent) if parent else id(root)
        grouped.setdefault(key, []).append(iframe)

    created = 0
    wrapped = 0

    for _, iframes in grouped.items():
        if not iframes:
            continue

        # Insert accordion before the first iframe in the group
        first = iframes[0]
        acc = _make_dp_accordion_from_iframes(soup, iframes, title_prefix="Panel")

        first.insert_before(acc)
        created += 1
        wrapped += len(iframes)

    # Add a spacer like your example (optional, but matches style)
    # We'll add <p>&nbsp;</p> after each accordion we created if not already present.
    for acc in root.find_all("div", class_=lambda c: c and "dp-panels-wrapper" in c):
        nxt = acc.find_next_sibling()
        if not (isinstance(nxt, Tag) and nxt.name == "p" and "&nbsp;" in str(nxt)):
            spacer = soup.new_tag("p")
            spacer.string = "\xa0"
            acc.insert_after(spacer)

    return str(root), {"wrapped_iframes": wrapped, "grouped_accordions_created": created}


# =============================================================================
# HTML CHUNKING
# =============================================================================

def split_html_into_chunks(html: str, max_chunk_chars: int = 7000) -> List[str]:
    html = (html or "").strip()
    if not html:
        return [""]

    soup = BeautifulSoup(html, "html.parser")
    root = soup.body if soup.body else soup

    nodes = []
    for n in list(root.contents):
        s = str(n)
        if s.strip():
            nodes.append(n)

    if not nodes:
        return [html]

    heading_names = {"h1", "h2", "h3", "h4", "h5", "h6"}
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush():
        nonlocal current_parts, current_len
        if current_parts:
            chunks.append("".join(current_parts).strip())
        current_parts = []
        current_len = 0

    for n in nodes:
        n_html = str(n)
        is_heading = getattr(n, "name", None) in heading_names
        if is_heading and current_parts:
            flush()
        if current_len + len(n_html) > max_chunk_chars and current_parts:
            flush()
        current_parts.append(n_html)
        current_len += len(n_html)

    flush()
    return chunks or [html]


# =============================================================================
# VALIDATION
# =============================================================================

_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}

LEGACY_CLASS_PATTERNS = [
    r"\bdesigntools\b",
    r"\bdt-",
    r"\bdesign-tools\b",
    r"\bcanvas-styler\b",
    r"\btoolkit\b",
]


def _strip_zero_width(s: str) -> str:
    for zw in _ZERO_WIDTH:
        s = s.replace(zw, "")
    return s


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _normalize_text_lenient(s: str) -> str:
    s = s or ""
    s = html_lib.unescape(s)
    s = s.replace("\xa0", " ")
    s = _strip_zero_width(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_attr_set(html: str, tag: str, attr: str) -> set[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    vals = set()
    for t in soup.find_all(tag):
        v = t.get(attr)
        if v:
            vals.add(str(v).strip())
    return vals


def _find_legacy_leakage(html: str) -> List[str]:
    """
    Return list of matched legacy patterns found in the HTML.
    """
    hits = []
    for pat in LEGACY_CLASS_PATTERNS:
        if re.search(pat, html, flags=re.IGNORECASE):
            hits.append(pat)
    return hits


def _all_iframes_in_dp_accordions(html: str) -> bool:
    soup = BeautifulSoup(html or "", "html.parser")
    root = soup.body if soup.body else soup
    for iframe in root.find_all("iframe"):
        if not _is_inside_dp_accordion(iframe):
            return False
    return True


def validate_rewrite(original_html: str, rewritten_html: str) -> Dict[str, Any]:
    o_text = _normalize_text_lenient(_visible_text(original_html))
    r_text = _normalize_text_lenient(_visible_text(rewritten_html))
    ok_text = (o_text == r_text)

    # Assets
    o_hrefs = _extract_attr_set(original_html, "a", "href")
    r_hrefs = _extract_attr_set(rewritten_html, "a", "href")
    o_srcs = _extract_attr_set(original_html, "img", "src") | _extract_attr_set(original_html, "source", "src")
    r_srcs = _extract_attr_set(rewritten_html, "img", "src") | _extract_attr_set(rewritten_html, "source", "src")
    o_iframes = _extract_attr_set(original_html, "iframe", "src")
    r_iframes = _extract_attr_set(rewritten_html, "iframe", "src")

    missing_hrefs = sorted(list(o_hrefs - r_hrefs))
    missing_srcs = sorted(list(o_srcs - r_srcs))
    missing_iframes = sorted(list(o_iframes - r_iframes))
    ok_assets = (len(missing_hrefs) == 0 and len(missing_srcs) == 0 and len(missing_iframes) == 0)

    # Structural rules
    legacy_hits = _find_legacy_leakage(rewritten_html)
    ok_no_legacy = (len(legacy_hits) == 0)

    ok_iframes_wrapped = _all_iframes_in_dp_accordions(rewritten_html)

    return {
        "ok": bool(ok_text and ok_assets and ok_no_legacy and ok_iframes_wrapped),
        "ok_text": ok_text,
        "ok_assets": ok_assets,
        "ok_no_legacy": ok_no_legacy,
        "ok_iframes_wrapped": ok_iframes_wrapped,
        "missing_hrefs": missing_hrefs[:50],
        "missing_srcs": missing_srcs[:50],
        "missing_iframes": missing_iframes[:50],
        "legacy_hits": legacy_hits,
    }


# =============================================================================
# STYLE GUIDE (FIX MASSIVE MODEL_CONTEXT) — DESIGNPLUS-ONLY
# =============================================================================

def build_style_guide_prompt(raw_model_context: str) -> str:
    raw_model_context = (raw_model_context or "").strip()
    return textwrap.dedent(
        f"""
        You are an expert Canvas + DesignPLUS HTML style analyst.

        TASK:
        Distill the following "model course/style examples" into a compact, actionable STYLE GUIDE.

        CRITICAL:
        - Extract ONLY DesignPLUS patterns.
        - If any legacy DesignTools patterns appear, IGNORE them and DO NOT include them.
        - The output guide must explicitly say "DesignPLUS only" and list common legacy tokens to avoid.

        HARD RULES:
        - Do NOT invent requirements that aren't supported by the model examples.
        - Keep it concise, but include concrete patterns and do/don’t rules.
        - Focus on structure, wrappers, DesignPLUS components, accessibility patterns.
        - Do NOT rewrite any educational text; this is style-only.
        - Output plain text (not HTML), max ~1200-1800 words.

        MODEL COURSE / STYLE EXAMPLES (raw):
        {raw_model_context}
        """
    ).strip()


def get_or_create_style_guide(client: OpenAI, raw_model_context: str) -> str:
    raw_model_context = (raw_model_context or "").strip()
    if not raw_model_context:
        return ""

    existing = st.session_state.get("style_guide", "")
    existing_key = st.session_state.get("style_guide_key", "")
    key = f"{len(raw_model_context)}:{hash(raw_model_context[:4000])}:{hash(raw_model_context[-4000:])}"

    if existing and existing_key == key:
        return existing

    if len(raw_model_context) <= 12000:
        style_guide = raw_model_context
        st.session_state["style_guide"] = style_guide
        st.session_state["style_guide_key"] = key
        return style_guide

    model_name = get_model_name()
    prompt = build_style_guide_prompt(raw_model_context)

    with st.spinner("Distilling model course into a compact DesignPLUS-only style guide…"):
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        style_guide = (resp.choices[0].message.content or "").strip()

    st.session_state["style_guide"] = style_guide
    st.session_state["style_guide_key"] = key
    return style_guide


# =============================================================================
# OPENAI REWRITE (CHUNKED) + POSTPROCESS ENFORCEMENT
# =============================================================================

def build_rewrite_prompt(
    item: Dict[str, Any],
    style_guide: str,
    global_instructions: str,
    html_fragment: str,
    original_visible_text: str,
    chunk_index: int,
    chunk_total: int,
) -> str:
    style_guide = (style_guide or "").strip()

    base_rules = textwrap.dedent(
        """
        You are an expert Canvas HTML editor.

        ABSOLUTE REQUIREMENTS (must follow):
        - Return ONLY HTML. No Markdown. No explanations.
        - DesignPLUS ONLY. Do NOT emit legacy DesignTools markup/classes.
        - Do NOT change, rewrite, paraphrase, reorder, summarize, or delete any visible text.
          The visible text content must remain EXACTLY the same as the input chunk.
        - Preserve all links (href), images (src), iframes (src), file links, IDs, anchors, and data-* attributes.
        - IMPORTANT: Do NOT attempt to invent custom accordion HTML for iframes.
          Iframes will be wrapped deterministically after rewrite. Keep iframe tags intact.
        - Focus on styling, structure, and accessibility only.
        """
    ).strip()

    item_type = item.get("type", "page")
    title = item.get("title", "")

    prompt = f"""
    {base_rules}

    GLOBAL INSTRUCTIONS (from user):
    {global_instructions or "Align structure and styling to the DesignPLUS style guide. Do not change visible text."}

    STYLE GUIDE / MODEL PATTERNS (DesignPLUS-only):
    {style_guide}

    TARGET ITEM:
    - Type: {item_type}
    - Title: {title}
    - Chunk: {chunk_index+1} of {chunk_total}

    ORIGINAL VISIBLE TEXT (MUST MATCH EXACTLY):
    {original_visible_text}

    ORIGINAL HTML (this chunk only):
    {html_fragment}

    OUTPUT:
    Rewrite ONLY this chunk's HTML to match the style guide and instructions.
    Return ONLY the rewritten HTML for this chunk.
    """.strip()

    return prompt


def _rewrite_chunk(client: OpenAI, model_name: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def rewrite_item_chunked_with_postprocess(
    client: OpenAI,
    item: Dict[str, Any],
    style_guide: str,
    global_instructions: str,
    max_chunk_chars: int = 7000,
) -> Tuple[str, Dict[str, Any]]:
    model_name = get_model_name()
    original_html = (item.get("original_html") or "").strip()
    if not original_html:
        return "", {"ok": True, "note": "No original HTML"}

    chunks = split_html_into_chunks(original_html, max_chunk_chars=max_chunk_chars)

    chunk_progress = st.progress(0, text=f"Rewriting '{item.get('title','')}'…")
    rewritten_chunks: List[str] = []

    for i, chunk_html in enumerate(chunks):
        original_visible = _normalize_text_lenient(_visible_text(chunk_html))
        prompt = build_rewrite_prompt(
            item=item,
            style_guide=style_guide,
            global_instructions=global_instructions,
            html_fragment=chunk_html,
            original_visible_text=original_visible,
            chunk_index=i,
            chunk_total=len(chunks),
        )
        out = _rewrite_chunk(client, model_name, prompt)
        rewritten_chunks.append(out)
        chunk_progress.progress((i + 1) / max(1, len(chunks)), text=f"Rewriting '{item.get('title','')}'… ({i+1}/{len(chunks)})")

    rewritten_html = "\n".join(rewritten_chunks).strip()

    # Deterministic enforcement: wrap iframes after rewrite
    rewritten_html, wrap_info = enforce_iframes_in_dp_accordions(rewritten_html)

    report = validate_rewrite(original_html, rewritten_html)
    report["iframe_wrap"] = wrap_info
    return rewritten_html, report


# =============================================================================
# STREAMLIT STATE INIT
# =============================================================================

for k, v in {
    "content_items": [],
    "model_context": "",
    "style_guide": "",
    "style_guide_key": "",
    "course_id": None,
    "rewrite_done": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Canvas Course Rewriter", layout="wide")
st.title("Canvas Course Rewriter (Streamlit)")

st.sidebar.header("Canvas connection")
target_course_id = st.sidebar.text_input(
    "Target course ID",
    help="Numeric ID from the Canvas course URL (e.g. .../courses/205033).",
)

if st.sidebar.button("Fetch course content"):
    if not target_course_id:
        st.sidebar.error("Please provide a target course ID.")
    else:
        base_url, token = get_canvas_config()
        try:
            with st.spinner("Fetching pages, assignments, and discussions from Canvas…"):
                _ = get_course(base_url, token, target_course_id)
                pages = get_pages(base_url, token, target_course_id)
                assignments = get_assignments(base_url, token, target_course_id)
                discussions = get_discussions(base_url, token, target_course_id)

                content_items: List[Dict[str, Any]] = []

                for p in pages:
                    content_items.append(
                        {
                            "type": "page",
                            "id": p["page_id"],
                            "canvas_id": p["page_id"],
                            "url_slug": p["url"],
                            "title": p["title"],
                            "original_html": p.get("body", "") or "",
                            "rewritten_html": "",
                            "approved": False,
                            "selected_for_rewrite": True,
                            "validation": {},
                        }
                    )

                for a in assignments:
                    content_items.append(
                        {
                            "type": "assignment",
                            "id": a["id"],
                            "canvas_id": a["id"],
                            "title": a["name"],
                            "original_html": a.get("description", "") or "",
                            "rewritten_html": "",
                            "approved": False,
                            "selected_for_rewrite": True,
                            "validation": {},
                        }
                    )

                for d in discussions:
                    content_items.append(
                        {
                            "type": "discussion",
                            "id": d["id"],
                            "canvas_id": d["id"],
                            "title": d["title"],
                            "original_html": d.get("message", "") or "",
                            "rewritten_html": "",
                            "approved": False,
                            "selected_for_rewrite": True,
                            "validation": {},
                        }
                    )

                st.session_state["content_items"] = content_items
                st.session_state["course_id"] = target_course_id
                st.session_state["rewrite_done"] = False

            st.success(f"Loaded {len(content_items)} items from course {target_course_id}.")
        except Exception as e:
            st.sidebar.error(f"Error fetching content: {e}")

# Step 2
st.header("Step 2 – Provide model course/style")

model_source = st.radio(
    "How do you want to provide a model?",
    ["Paste HTML/JSON", "Upload a file", "Use Canvas model course"],
    horizontal=True,
)

if model_source == "Paste HTML/JSON":
    pasted = st.text_area("Paste HTML/JSON describing model course/style:", height=200, key="pasted_model")
    if st.button("Use this as model"):
        st.session_state["model_context"] = pasted or ""
        st.session_state["style_guide"] = ""
        st.session_state["style_guide_key"] = ""
        st.success("Model context updated.")

elif model_source == "Upload a file":
    uploaded = st.file_uploader("Upload an HTML/JSON/TXT file for model style.", type=["html", "htm", "json", "txt"])
    if uploaded is not None and st.button("Use uploaded file as model"):
        content = uploaded.read().decode("utf-8", errors="ignore")
        st.session_state["model_context"] = content
        st.session_state["style_guide"] = ""
        st.session_state["style_guide_key"] = ""
        st.success("Model context loaded.")

elif model_source == "Use Canvas model course":
    model_course_id = st.text_input("Model course ID (numeric)", key="model_course_id")
    max_model_items = st.number_input("Max model items to pull", min_value=3, max_value=50, value=10, step=1)
    if st.button("Fetch model course content"):
        if not model_course_id:
            st.error("Model course ID is required.")
        else:
            base_url, token = get_canvas_config()
            try:
                with st.spinner("Fetching model course content…"):
                    pages_m = get_pages(base_url, token, model_course_id, max_items=max_model_items)
                    assignments_m = get_assignments(base_url, token, model_course_id, max_items=max_model_items)
                    discussions_m = get_discussions(base_url, token, model_course_id, max_items=max_model_items)

                    model_snips = []
                    for p in pages_m[:max_model_items]:
                        model_snips.append(f"### [page] {p['title']}\n{p.get('body', '')}")
                    for a in assignments_m[:max_model_items]:
                        model_snips.append(f"### [assignment] {a['name']}\n{a.get('description', '')}")
                    for d in discussions_m[:max_model_items]:
                        model_snips.append(f"### [discussion] {d['title']}\n{d.get('message', '')}")

                    st.session_state["model_context"] = "\n\n".join(model_snips)
                    st.session_state["style_guide"] = ""
                    st.session_state["style_guide_key"] = ""

                st.success("Model context built from Canvas model course.")
            except Exception as e:
                st.error(f"Error fetching model course: {e}")

if st.session_state["model_context"]:
    with st.expander("Preview model context (trimmed)", expanded=False):
        st.text_area("Model context preview:", value=st.session_state["model_context"][:4000], height=200)

# Step 3
st.header("Step 3 – Select items and rewrite")

global_instructions = st.text_area(
    "High-level rewrite instructions (style/structure only):",
    placeholder="Standardize layout, apply DesignPLUS patterns, improve accessibility structure, etc.",
    height=150,
    key="global_instructions",
)

items = st.session_state["content_items"]
st.subheader("Choose which items to rewrite")

if items:
    colf1, colf2, colf3, colf4 = st.columns([1, 1, 1, 2])
    with colf1:
        filter_pages = st.checkbox("Pages", value=True)
    with colf2:
        filter_assignments = st.checkbox("Assignments", value=True)
    with colf3:
        filter_discussions = st.checkbox("Discussions", value=True)
    with colf4:
        search = st.text_input("Search titles", value="")

    cola, colb, _ = st.columns([1, 1, 3])
    with cola:
        if st.button("Select all shown"):
            for it in items:
                if ((it["type"] == "page" and filter_pages) or
                    (it["type"] == "assignment" and filter_assignments) or
                    (it["type"] == "discussion" and filter_discussions)):
                    if not search or search.lower() in it.get("title", "").lower():
                        it["selected_for_rewrite"] = True
            st.session_state["content_items"] = items

    with colb:
        if st.button("Select none shown"):
            for it in items:
                if ((it["type"] == "page" and filter_pages) or
                    (it["type"] == "assignment" and filter_assignments) or
                    (it["type"] == "discussion" and filter_discussions)):
                    if not search or search.lower() in it.get("title", "").lower():
                        it["selected_for_rewrite"] = False
            st.session_state["content_items"] = items

    shown = 0
    for idx, it in enumerate(items):
        if it["type"] == "page" and not filter_pages:
            continue
        if it["type"] == "assignment" and not filter_assignments:
            continue
        if it["type"] == "discussion" and not filter_discussions:
            continue
        if search and search.lower() not in it.get("title", "").lower():
            continue

        shown += 1
        it["selected_for_rewrite"] = st.checkbox(
            f"[{it['type']}] {it.get('title', '')}",
            value=bool(it.get("selected_for_rewrite", True)),
            key=f"rewrite_pick_{idx}",
        )
    st.caption(f"Showing {shown} items. Only selected items will be sent to the LLM.")
else:
    st.info("Load course content first using the sidebar.")

with st.expander("Advanced rewrite settings", expanded=False):
    max_chunk_chars = st.slider("Max characters per HTML chunk", 2000, 12000, 7000, 500)
    show_validation = st.checkbox("Show validation details per item", value=True)

can_run_rewrite = bool(st.session_state["content_items"] and st.session_state["model_context"])

if st.button("Run rewrite on selected items", disabled=not can_run_rewrite):
    selected_items = [it for it in st.session_state["content_items"] if it.get("selected_for_rewrite", True)]
    skipped_count = len(st.session_state["content_items"]) - len(selected_items)

    if not selected_items:
        st.warning("No items selected for rewrite.")
        st.stop()

    client = get_ai_client()
    style_guide = get_or_create_style_guide(client, st.session_state["model_context"])

    progress = st.progress(0.0)
    status_area = st.empty()

    for idx, item in enumerate(selected_items):
        status_area.write(f"Rewriting [{item['type']}] {item['title']}…")
        try:
            rewritten, report = rewrite_item_chunked_with_postprocess(
                client=client,
                item=item,
                style_guide=style_guide,
                global_instructions=global_instructions,
                max_chunk_chars=max_chunk_chars,
            )
            item["rewritten_html"] = rewritten
            item["validation"] = report
            item.pop("rewrite_error", None)
        except Exception as e:
            item["rewrite_error"] = str(e)
            item["validation"] = {"ok": False, "error": str(e)}

        progress.progress((idx + 1) / len(selected_items))

    st.session_state["rewrite_done"] = True
    status_area.write(f"Rewrite complete. Rewrote {len(selected_items)} item(s), skipped {skipped_count}.")

# Step 4
st.header("Step 4 – Review and approve changes")

items = st.session_state["content_items"]

if not items:
    st.info("Load course content first using the sidebar.")
else:
    for i, item in enumerate(items):
        has_rewrite = bool(item.get("rewritten_html"))
        label = f"[{item['type']}] {item['title']}"
        with st.expander(label, expanded=False):
            if not item.get("selected_for_rewrite", True) and not has_rewrite:
                st.info("This item is currently not selected for rewrite (skipped).")

            if item.get("rewrite_error"):
                st.error(f"Rewrite error: {item['rewrite_error']}")

            val = item.get("validation") or {}
            if show_validation and val:
                if val.get("ok", False):
                    st.success("Validator: OK (text preserved + assets preserved + DesignPLUS-only + iframes wrapped).")
                    if val.get("iframe_wrap"):
                        st.caption(f"Iframe wrap: {val['iframe_wrap']}")
                else:
                    st.warning("Validator: FAILED")
                    if val.get("ok_text") is False:
                        st.write("- Visible text mismatch")
                    if val.get("missing_hrefs"):
                        st.write(f"- Missing href(s): {val.get('missing_hrefs')[:10]}")
                    if val.get("missing_srcs"):
                        st.write(f"- Missing src(s): {val.get('missing_srcs')[:10]}")
                    if val.get("missing_iframes"):
                        st.write(f"- Missing iframe src(s): {val.get('missing_iframes')[:10]}")
                    if val.get("ok_no_legacy") is False:
                        st.write(f"- Legacy markup detected: {val.get('legacy_hits')}")
                    if val.get("ok_iframes_wrapped") is False:
                        st.write("- Some iframes are NOT inside a dp-panels-wrapper accordion")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original (visual)")
                if item.get("original_html"):
                    components.html(item["original_html"], height=350, scrolling=True)
                else:
                    st.info("No HTML body for this item.")

            with col2:
                st.subheader("Proposed (visual)")
                if has_rewrite:
                    components.html(item["rewritten_html"], height=350, scrolling=True)
                else:
                    st.warning("No rewrite available yet. Run the rewrite step above.")

            approved = st.checkbox("Approve this change", value=item.get("approved", False), key=f"approved_{i}")
            item["approved"] = approved

    st.session_state["content_items"] = items

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Approve ALL items with proposed HTML"):
            for it in st.session_state["content_items"]:
                if it.get("rewritten_html"):
                    it["approved"] = True
            st.success("All items with proposed HTML marked as approved.")

    with col_b:
        if st.button("Clear ALL approvals"):
            for it in st.session_state["content_items"]:
                it["approved"] = False
            st.info("All approvals cleared.")

# Step 5
st.header("Step 5 – Write approved changes back to Canvas")

if st.button("Write approved changes to Canvas"):
    if not st.session_state["course_id"]:
        st.error("Target course ID is missing (use the sidebar to load a course).")
    else:
        base_url, token = get_canvas_config()
        course_id = st.session_state["course_id"]
        approved_items = [it for it in st.session_state["content_items"] if it.get("approved") and it.get("rewritten_html")]

        if not approved_items:
            st.warning("No approved items with rewritten HTML to write back.")
        else:
            with st.spinner(f"Writing {len(approved_items)} items back to Canvas…"):
                errors = []
                for item in approved_items:
                    try:
                        if item["type"] == "page":
                            update_page_html(base_url, token, course_id, item["url_slug"], item["rewritten_html"])
                        elif item["type"] == "assignment":
                            update_assignment_html(base_url, token, course_id, item["canvas_id"], item["rewritten_html"])
                        elif item["type"] == "discussion":
                            update_discussion_html(base_url, token, course_id, item["canvas_id"], item["rewritten_html"])
                    except Exception as e:
                        errors.append((item["title"], str(e)))

            if errors:
                st.error("Some items failed to update:")
                for title, msg in errors:
                    st.write(f"- **{title}**: {msg}")
            else:
                st.success("All approved items successfully written back to Canvas.")
