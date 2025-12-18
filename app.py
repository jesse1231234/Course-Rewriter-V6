import os
import re
import textwrap
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from bs4 import BeautifulSoup


# =============================================================================
# CONFIG / CLIENTS
# =============================================================================

def get_ai_client() -> OpenAI:
    """
    Create an OpenAI client for an Azure AI Foundry / Project endpoint.

    Required config (Streamlit secrets OR env vars):
      - OPENAI_BASE_URL  e.g. "https://<something>.services.ai.azure.com/openai/v1"
      - OPENAI_API_KEY   e.g. the key from the Foundry 'Use this model' / 'Connections' blade
      - OPENAI_MODEL     e.g. deployment name in Azure (Deployment Info → Name)
    """
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

        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1 : part.find(">")]
                break
        url = next_url
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

        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1 : part.find(">")]
                break
        url = next_url
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

        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1 : part.find(">")]
                break
        url = next_url
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
# HTML CHUNKING + VALIDATION
# =============================================================================

def split_html_into_chunks(html: str, max_chunk_chars: int = 7000) -> List[str]:
    """
    Split HTML into chunks without cutting through tags.

    Strategy:
      - Parse into a soup fragment
      - Walk top-level nodes
      - Prefer starting a new chunk at headings (h1-h6)
      - Enforce approximate max char size by grouping nodes
    """
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


def _normalize_visible_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")

    # Remove non-visible content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_attr_set(html: str, tag: str, attr: str) -> set[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    vals = set()
    for t in soup.find_all(tag):
        v = t.get(attr)
        if v:
            vals.add(str(v).strip())
    return vals


def validate_rewrite(original_html: str, rewritten_html: str) -> Dict[str, Any]:
    """
    Validate that:
      1) Visible text is identical (your "no content edits" requirement)
      2) Core media/link attributes aren't lost (href/src/iframe src)
    Returns a dict with ok flag + details.
    """
    o_text = _normalize_visible_text(original_html)
    r_text = _normalize_visible_text(rewritten_html)

    ok_text = (o_text == r_text)

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

    return {
        "ok": bool(ok_text and ok_assets),
        "ok_text": ok_text,
        "ok_assets": ok_assets,
        "missing_hrefs": missing_hrefs[:50],
        "missing_srcs": missing_srcs[:50],
        "missing_iframes": missing_iframes[:50],
    }


# =============================================================================
# STYLE GUIDE (FIX MASSIVE MODEL_CONTEXT)
# =============================================================================

def build_style_guide_prompt(raw_model_context: str) -> str:
    raw_model_context = (raw_model_context or "").strip()

    return textwrap.dedent(
        f"""
        You are an expert Canvas + DesignPLUS HTML style analyst.

        TASK:
        Distill the following "model course/style examples" into a compact, actionable STYLE GUIDE
        that another assistant can follow when rewriting Canvas HTML.

        HARD RULES:
        - Do NOT invent requirements that aren't supported by the model examples.
        - Keep it concise, but include concrete patterns and “do/don’t” rules.
        - Focus on structure, wrappers, DesignPLUS components, CSU branding usage, accessibility patterns.
        - Do NOT rewrite any educational text; this is style-only.
        - Output plain text (not HTML), max ~1200-1800 words.

        MODEL COURSE / STYLE EXAMPLES (raw):
        {raw_model_context}
        """
    ).strip()


def get_or_create_style_guide(client: OpenAI, raw_model_context: str) -> str:
    """
    If raw model context is huge, create a compact style guide once and reuse it for every chunk.
    Stored in session_state["style_guide"].
    """
    raw_model_context = (raw_model_context or "").strip()
    if not raw_model_context:
        return ""

    # If already created for this exact raw context, reuse
    existing = st.session_state.get("style_guide", "")
    existing_key = st.session_state.get("style_guide_key", "")

    # a cheap fingerprint to avoid re-creating every run
    key = f"{len(raw_model_context)}:{hash(raw_model_context[:4000])}:{hash(raw_model_context[-4000:])}"

    if existing and existing_key == key:
        return existing

    # If small enough, just use raw (but still capped) to avoid extra call
    # (You can lower this threshold if you want.)
    if len(raw_model_context) <= 12000:
        style_guide = raw_model_context
        st.session_state["style_guide"] = style_guide
        st.session_state["style_guide_key"] = key
        return style_guide

    # Otherwise, distill with a single model call
    model_name = get_model_name()
    prompt = build_style_guide_prompt(raw_model_context)

    with st.spinner("Distilling model course into a compact style guide…"):
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
# OPENAI REWRITE (CHUNKED + VALIDATOR PASS)
# =============================================================================

def build_rewrite_prompt(
    item: Dict[str, Any],
    style_guide: str,
    global_instructions: str,
    html_fragment: str,
    chunk_index: int,
    chunk_total: int,
    repair_notes: Optional[str] = None,
) -> str:
    style_guide = (style_guide or "").strip()

    base_rules = textwrap.dedent(
        """
        You are an expert Canvas HTML editor.

        HARD REQUIREMENTS (must follow):
        - Return ONLY HTML. No Markdown. No explanations.
        - Do NOT change, rewrite, paraphrase, reorder, summarize, or delete any visible text.
          The visible text content must remain EXACTLY the same as the input chunk.
        - You MAY adjust HTML structure/wrappers/classes and add DesignPLUS structure needed for styling/accessibility,
          but you must not alter the text itself.
        - Preserve all links (href), images (src), iframes (src), file links, IDs, anchors, and data-* attributes.
        - Use DesignPLUS styling/patterns consistent with the provided style guide / model course.
        - Use Colorado State University branding colors where appropriate (style-only).
        - Place all iframes within DesignPLUS accordions.
        - Focus on styling, structure, and accessibility only.
        """
    ).strip()

    item_type = item.get("type", "page")
    title = item.get("title", "")

    repair_block = ""
    if repair_notes:
        repair_block = textwrap.dedent(
            f"""
            VALIDATION FAILURES TO FIX (do not ignore):
            {repair_notes}
            """
        ).strip()

    prompt = f"""
    {base_rules}

    GLOBAL INSTRUCTIONS (from user):
    {global_instructions or "Align structure and styling to the model style guide. Do not change visible text."}

    STYLE GUIDE / MODEL PATTERNS (condensed):
    {style_guide}

    TARGET ITEM:
    - Type: {item_type}
    - Title: {title}
    - Chunk: {chunk_index+1} of {chunk_total}

    {repair_block}

    ORIGINAL HTML (this chunk only):
    {html_fragment}

    OUTPUT:
    Rewrite ONLY this chunk's HTML to match the style guide and global instructions.
    Return ONLY the rewritten HTML for this chunk.
    """.strip()

    return prompt


def _rewrite_chunk(
    client: OpenAI,
    model_name: str,
    prompt: str,
) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


def rewrite_item_chunked_with_validation(
    client: OpenAI,
    item: Dict[str, Any],
    style_guide: str,
    global_instructions: str,
    max_chunk_chars: int = 7000,
    max_repair_passes: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    """
    1) Chunk the original HTML
    2) Rewrite each chunk
    3) Reassemble
    4) Validate no visible text changes + no asset loss
    5) If fails, do a repair pass (re-run chunking and only repair failing chunks)
    Returns (rewritten_html, validation_report)
    """
    model_name = get_model_name()

    original_html = (item.get("original_html") or "").strip()
    if not original_html:
        return "", {"ok": True, "ok_text": True, "ok_assets": True, "note": "No original HTML"}

    chunks = split_html_into_chunks(original_html, max_chunk_chars=max_chunk_chars)

    # Progress UI per item
    chunk_progress = st.progress(0, text=f"Rewriting '{item.get('title','')}'…")
    rewritten_chunks: List[str] = []
    for i, chunk_html in enumerate(chunks):
        prompt = build_rewrite_prompt(
            item=item,
            style_guide=style_guide,
            global_instructions=global_instructions,
            html_fragment=chunk_html,
            chunk_index=i,
            chunk_total=len(chunks),
        )
        out = _rewrite_chunk(client, model_name, prompt)
        rewritten_chunks.append(out)
        chunk_progress.progress((i + 1) / max(1, len(chunks)), text=f"Rewriting '{item.get('title','')}'… ({i+1}/{len(chunks)})")

    rewritten_html = "\n".join(rewritten_chunks).strip()
    report = validate_rewrite(original_html, rewritten_html)

    # Optional repair passes (target failing chunks)
    passes = 0
    while not report["ok"] and passes < max_repair_passes:
        passes += 1

        # Figure out which chunks are failing the "text unchanged" check
        failing_chunk_idxs: List[int] = []
        for i, chunk_html in enumerate(chunks):
            # Validate chunk-by-chunk for tighter repair targeting
            r_chunk = rewritten_chunks[i] if i < len(rewritten_chunks) else ""
            c_report = validate_rewrite(chunk_html, r_chunk)
            if not c_report["ok"]:
                failing_chunk_idxs.append(i)

        # If we can't isolate, repair all
        if not failing_chunk_idxs:
            failing_chunk_idxs = list(range(len(chunks)))

        # Build a compact repair note
        repair_notes = []
        if not report.get("ok_text", True):
            repair_notes.append("- Visible text changed. Restore EXACT original visible text.")
        if report.get("missing_hrefs"):
            repair_notes.append(f"- Missing href(s): {report['missing_hrefs'][:10]}")
        if report.get("missing_srcs"):
            repair_notes.append(f"- Missing src(s): {report['missing_srcs'][:10]}")
        if report.get("missing_iframes"):
            repair_notes.append(f"- Missing iframe src(s): {report['missing_iframes'][:10]}")
        repair_notes_str = "\n".join(repair_notes).strip() or "- Validation failed. Fix issues without changing visible text."

        # Repair only failing chunks
        for i in failing_chunk_idxs:
            chunk_html = chunks[i]
            prompt = build_rewrite_prompt(
                item=item,
                style_guide=style_guide,
                global_instructions=global_instructions,
                html_fragment=chunk_html,
                chunk_index=i,
                chunk_total=len(chunks),
                repair_notes=repair_notes_str,
            )
            rewritten_chunks[i] = _rewrite_chunk(client, model_name, prompt)

        rewritten_html = "\n".join(rewritten_chunks).strip()
        report = validate_rewrite(original_html, rewritten_html)
        report["repair_passes_used"] = passes
        report["repaired_chunk_count"] = len(failing_chunk_idxs)

    return rewritten_html, report


# =============================================================================
# STREAMLIT STATE INIT
# =============================================================================

if "content_items" not in st.session_state:
    st.session_state["content_items"] = []

if "model_context" not in st.session_state:
    st.session_state["model_context"] = ""

if "style_guide" not in st.session_state:
    st.session_state["style_guide"] = ""

if "style_guide_key" not in st.session_state:
    st.session_state["style_guide_key"] = ""

if "course_id" not in st.session_state:
    st.session_state["course_id"] = None

if "rewrite_done" not in st.session_state:
    st.session_state["rewrite_done"] = False


# =============================================================================
# UI: SIDEBAR
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
                            "validation": {},
                        }
                    )

                st.session_state["content_items"] = content_items
                st.session_state["course_id"] = target_course_id
                st.session_state["rewrite_done"] = False

            st.success(f"Loaded {len(content_items)} items from course {target_course_id}.")
        except Exception as e:
            st.sidebar.error(f"Error fetching content: {e}")


# =============================================================================
# STEP 2: MODEL INPUT
# =============================================================================

st.header("Step 2 – Provide model course/style")

model_source = st.radio(
    "How do you want to provide a model?",
    ["Paste HTML/JSON", "Upload a file", "Use Canvas model course"],
    horizontal=True,
)

if model_source == "Paste HTML/JSON":
    pasted = st.text_area(
        "Paste HTML, JSON, or other structured description of your model course/style:",
        height=200,
        key="pasted_model",
    )
    if st.button("Use this as model"):
        st.session_state["model_context"] = pasted or ""
        st.session_state["style_guide"] = ""  # invalidate cached guide
        st.session_state["style_guide_key"] = ""
        st.success("Model context updated from pasted content.")

elif model_source == "Upload a file":
    uploaded = st.file_uploader(
        "Upload an HTML / JSON / TXT file that represents your model course/style.",
        type=["html", "htm", "json", "txt"],
    )
    if uploaded is not None and st.button("Use uploaded file as model"):
        content = uploaded.read().decode("utf-8", errors="ignore")
        st.session_state["model_context"] = content
        st.session_state["style_guide"] = ""  # invalidate cached guide
        st.session_state["style_guide_key"] = ""
        st.success("Model context loaded from uploaded file.")

elif model_source == "Use Canvas model course":
    model_course_id = st.text_input("Model course ID (numeric, from Canvas URL)", key="model_course_id")
    max_model_items = st.number_input(
        "Max items to pull from model course (total across types)",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )
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

                    model_snippets = []

                    for p in pages_m[:max_model_items]:
                        model_snippets.append(f"### [page] {p['title']}\n{p.get('body', '')}")

                    for a in assignments_m[:max_model_items]:
                        model_snippets.append(f"### [assignment] {a['name']}\n{a.get('description', '')}")

                    for d in discussions_m[:max_model_items]:
                        model_snippets.append(f"### [discussion] {d['title']}\n{d.get('message', '')}")

                    st.session_state["model_context"] = "\n\n".join(model_snippets)
                    st.session_state["style_guide"] = ""  # invalidate cached guide
                    st.session_state["style_guide_key"] = ""

                st.success("Model context built from Canvas model course.")
            except Exception as e:
                st.error(f"Error fetching model course: {e}")


if st.session_state["model_context"]:
    with st.expander("Preview current model context (trimmed)", expanded=False):
        st.text_area(
            "Model context preview:",
            value=st.session_state["model_context"][:4000],
            height=200,
        )


# =============================================================================
# STEP 3: CONFIGURE & RUN REWRITE
# =============================================================================

st.header("Step 3 – Rewrite course content with Azure OpenAI")

global_instructions = st.text_area(
    "High-level rewrite instructions (style/structure only):",
    placeholder="E.g., standardize headings, apply CSU Online DesignPLUS layout, wrap embeds in accordions, improve accessibility structure, etc.",
    height=150,
    key="global_instructions",
)

with st.expander("Advanced rewrite settings", expanded=False):
    max_chunk_chars = st.slider("Max characters per HTML chunk", 2000, 12000, 7000, 500)
    repair_passes = st.slider("Validator repair passes", 0, 2, 1, 1)
    show_validation = st.checkbox("Show validation details per item", value=True)

can_run_rewrite = bool(st.session_state["content_items"] and st.session_state["model_context"])

if st.button("Run rewrite on all items", disabled=not can_run_rewrite):
    client = get_ai_client()
    items = st.session_state["content_items"]
    raw_model_context = st.session_state["model_context"]

    # FIX: massive model_context -> compact style guide once, reused everywhere
    style_guide = get_or_create_style_guide(client, raw_model_context)

    progress = st.progress(0.0)
    status_area = st.empty()

    for idx, item in enumerate(items):
        status_area.write(f"Rewriting [{item['type']}] {item['title']}…")
        try:
            if item.get("original_html"):
                rewritten, report = rewrite_item_chunked_with_validation(
                    client=client,
                    item=item,
                    style_guide=style_guide,
                    global_instructions=global_instructions,
                    max_chunk_chars=max_chunk_chars,
                    max_repair_passes=repair_passes,
                )
                item["rewritten_html"] = rewritten
                item["validation"] = report
            else:
                item["rewritten_html"] = ""
                item["validation"] = {"ok": True, "note": "No original HTML"}
        except Exception as e:
            item["rewrite_error"] = str(e)
            item["validation"] = {"ok": False, "error": str(e)}

        progress.progress((idx + 1) / len(items))

    st.session_state["content_items"] = items
    st.session_state["rewrite_done"] = True
    status_area.write("Rewrite complete.")


# =============================================================================
# STEP 4: REVIEW & APPROVAL
# =============================================================================

st.header("Step 4 – Review and approve changes")

items = st.session_state["content_items"]

if not items:
    st.info("Load course content first using the sidebar.")
else:
    for i, item in enumerate(items):
        has_rewrite = bool(item.get("rewritten_html"))
        label = f"[{item['type']}] {item['title']}"
        with st.expander(label, expanded=False):
            val = item.get("validation") or {}
            if item.get("rewrite_error"):
                st.error(f"Rewrite error: {item['rewrite_error']}")

            if show_validation and val:
                if val.get("ok", False):
                    st.success("Validator: OK (text preserved + assets preserved).")
                else:
                    st.warning("Validator: FAILED (see details below).")
                    if val.get("ok_text") is False:
                        st.write("- Visible text mismatch")
                    if val.get("missing_hrefs"):
                        st.write(f"- Missing href(s): {val.get('missing_hrefs')[:10]}")
                    if val.get("missing_srcs"):
                        st.write(f"- Missing src(s): {val.get('missing_srcs')[:10]}")
                    if val.get("missing_iframes"):
                        st.write(f"- Missing iframe src(s): {val.get('missing_iframes')[:10]}")
                    if val.get("repair_passes_used"):
                        st.caption(f"Repair passes used: {val['repair_passes_used']}")

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
                    st.caption("Proposed version based on style guide + instructions.")
                else:
                    st.warning("No rewrite available yet. Run the rewrite step above.")

            approved = st.checkbox(
                "Approve this change",
                value=item.get("approved", False),
                key=f"approved_{i}",
            )
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


# =============================================================================
# STEP 5: WRITE BACK TO CANVAS
# =============================================================================

st.header("Step 5 – Write approved changes back to Canvas")

if st.button("Write approved changes to Canvas"):
    if not st.session_state["course_id"]:
        st.error("Target course ID is missing (use the sidebar to load a course).")
    else:
        base_url, token = get_canvas_config()
        course_id = st.session_state["course_id"]
        approved_items = [
            it for it in st.session_state["content_items"]
            if it.get("approved") and it.get("rewritten_html")
        ]

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
