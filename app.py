import os
import re
import json
import html as html_lib
import textwrap
from dataclasses import dataclass, asdict
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
        st.error("Missing OPENAI_BASE_URL / OPENAI_API_KEY in secrets or env.")
        st.stop()
    return OpenAI(base_url=base_url, api_key=api_key)


def get_model_name() -> str:
    model_name = st.secrets.get("OPENAI_MODEL", None) or os.getenv("OPENAI_MODEL")
    if not model_name:
        st.error("Missing OPENAI_MODEL in secrets or env (Azure deployment name).")
        st.stop()
    return model_name


def get_canvas_config() -> tuple[str, str]:
    base_url = st.secrets.get("CANVAS_BASE_URL", None) or os.getenv("CANVAS_BASE_URL")
    token = st.secrets.get("CANVAS_API_TOKEN", None) or os.getenv("CANVAS_API_TOKEN")
    if not base_url or not token:
        st.error("Missing CANVAS_BASE_URL / CANVAS_API_TOKEN in secrets or env.")
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
# TEXT UTILS
# =============================================================================

_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}

def _strip_zero_width(s: str) -> str:
    for zw in _ZERO_WIDTH:
        s = s.replace(zw, "")
    return s


def visible_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def norm_text(s: str) -> str:
    s = s or ""
    s = html_lib.unescape(s)
    s = s.replace("\xa0", " ")
    s = _strip_zero_width(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preview_text(html: str, max_chars: int = 700) -> str:
    t = norm_text(visible_text(html))
    return t if len(t) <= max_chars else (t[:max_chars].rstrip() + "…")


# =============================================================================
# STYLE CONTRACT (USER-CONTROLLED)
# =============================================================================

@dataclass
class StyleContract:
    # Wrapper / theme
    wrapper_variant: str  # CSS classes for dp-wrapper
    header_enabled: bool
    header_variant: str   # CSS classes for header
    header_title_source: str  # "canvas_title" | "llm_suggest" | "custom"
    header_title_custom: str

    # Content block
    data_category: str

    # Accordion / embeds
    wrap_iframes: bool
    iframe_grouping: str  # "group_consecutive" | "one_per_iframe"
    panel_title_source: str  # "iframe_title" | "preceding_text_llm" | "generic"
    panel_title_generic_prefix: str

    # Guardrails
    remove_legacy: bool


# A few sane presets; you can add more as you learn what you like.
WRAPPER_PRESETS = {
    "Flat Sections (variation-2)": "dp-wrapper dp-flat-sections-main variation-2",
    "Circle Left + Flat Sections (variation-2)": "dp-wrapper dp-circle-left dp-flat-sections-main variation-2",
    "Module Header + Flat Sections (variation-2)": "dp-wrapper dp-flat-sections-main variation-2",
}

HEADER_PRESETS = {
    "Circle Left": "dp-header dp-header-circle-left",
    "Module": "dp-header dp-header-module",
    "Basic": "dp-header",
}


LOCKED_TOKEN = "{{LOCKED_CONTENT}}"


# =============================================================================
# DETERMINISTIC CLEANUP: ban legacy / kl_ panels
# =============================================================================

LEGACY_PATTERNS = [
    r"\bdesigntools\b",
    r"\bdt-",
    r"\bdesign-tools\b",
    r"\bkl_panels_wrapper\b",
    r"\bkl_panel\b",
]

def strip_kl_and_legacy(html: str) -> Tuple[str, Dict[str, Any]]:
    soup = BeautifulSoup(html or "", "html.parser")
    root = soup.body if soup.body else soup

    removed = 0

    # Remove any nodes with class starting with kl_
    for tag in list(root.find_all(True)):
        cls = tag.get("class", [])
        if isinstance(cls, list) and any(str(c).startswith("kl_") for c in cls):
            tag.decompose()
            removed += 1

    # Remove kl_panels_wrapper blocks specifically (belt & suspenders)
    for div in root.find_all("div", class_=lambda c: c and "kl_panels_wrapper" in str(c)):
        div.decompose()
        removed += 1

    return str(root), {"removed_nodes": removed}


# =============================================================================
# CANONICAL DESIGNPLUS ACCORDION (DETERMINISTIC)
# =============================================================================

DP_ACCORDION_WRAPPER_CLASS = "dp-panels-wrapper dp-accordion-default"
DP_PANEL_GROUP_CLASS = "dp-panel-group"
DP_PANEL_HEADING_CLASS = "dp-panel-heading"
DP_PANEL_CONTENT_CLASS = "dp-panel-content"
DP_EMBED_WRAPPER_CLASS = "dp-embed-wrapper"

def _is_inside_dp_accordion(tag: Tag) -> bool:
    p = tag.parent
    while p is not None:
        if isinstance(p, Tag):
            cls = p.get("class", [])
            cls_str = " ".join(cls) if isinstance(cls, list) else str(cls)
            if "dp-panels-wrapper" in cls_str:
                return True
        p = p.parent
    return False


def _make_dp_accordion(soup: BeautifulSoup, iframes: List[Tag], panel_titles: List[str]) -> Tag:
    wrapper = soup.new_tag("div")
    wrapper["class"] = DP_ACCORDION_WRAPPER_CLASS.split()

    for idx, iframe in enumerate(iframes):
        panel_group = soup.new_tag("div")
        panel_group["class"] = [DP_PANEL_GROUP_CLASS]

        heading = soup.new_tag("h3")
        heading["class"] = [DP_PANEL_HEADING_CLASS]
        heading.string = panel_titles[idx] if idx < len(panel_titles) else f"Panel {idx+1}"

        content = soup.new_tag("div")
        content["class"] = [DP_PANEL_CONTENT_CLASS]

        embed = soup.new_tag("div")
        embed["class"] = [DP_EMBED_WRAPPER_CLASS]

        iframe.extract()
        embed.append(iframe)
        content.append(embed)

        panel_group.append(heading)
        panel_group.append(content)
        wrapper.append(panel_group)

    return wrapper


def _group_consecutive_iframes(root: Tag) -> List[List[Tag]]:
    """
    Group iframes that appear consecutively (ignoring whitespace text nodes),
    which matches common "Videos:" sections that are a stack of iframes.
    """
    groups: List[List[Tag]] = []
    current: List[Tag] = []

    # We iterate through descendants in document order but only consider top-level-ish flow:
    # Approach: scan all iframes and group by nearest block parent id.
    # Simpler + robust: group by iframe parent (common case) and adjacency.
    # We'll do adjacency by previous sibling chain.
    orphans = [i for i in root.find_all("iframe") if not _is_inside_dp_accordion(i)]
    if not orphans:
        return []

    # Group by shared parent first
    by_parent: Dict[int, List[Tag]] = {}
    for iframe in orphans:
        p = iframe.parent
        key = id(p) if p else id(root)
        by_parent.setdefault(key, []).append(iframe)

    for _, iframes in by_parent.items():
        # preserve document order
        iframes = sorted(iframes, key=lambda x: x.sourceline or 0)
        # now group adjacency by checking if next iframe is "near" in DOM
        # We’ll treat them as one group if they share the same parent (already) and are close.
        # For safety, just keep them in one group; it matches your common video stacks.
        groups.append(iframes)

    return groups


def enforce_iframe_accordions(
    html: str,
    contract: StyleContract,
    llm_panel_titles: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Wrap orphan iframes into canonical DP accordions.
    Panel titles are determined by contract:
      - iframe_title
      - preceding_text_llm (uses llm_panel_titles mapping src->title)
      - generic
    """
    soup = BeautifulSoup(html or "", "html.parser")
    root = soup.body if soup.body else soup

    if not contract.wrap_iframes:
        return str(root), {"wrapped_iframes": 0, "accordions_created": 0}

    # Identify orphan iframes
    orphans = [i for i in root.find_all("iframe") if not _is_inside_dp_accordion(i)]
    if not orphans:
        return str(root), {"wrapped_iframes": 0, "accordions_created": 0}

    if contract.iframe_grouping == "group_consecutive":
        groups = _group_consecutive_iframes(root)
    else:
        groups = [[i] for i in orphans]

    created = 0
    wrapped = 0

    for group in groups:
        if not group:
            continue

        titles: List[str] = []
        for idx, iframe in enumerate(group):
            src = (iframe.get("src") or "").strip()
            iframe_title = (iframe.get("title") or "").strip()

            if contract.panel_title_source == "iframe_title" and iframe_title:
                titles.append(iframe_title)
            elif contract.panel_title_source == "preceding_text_llm" and llm_panel_titles and src in llm_panel_titles:
                titles.append(llm_panel_titles[src])
            elif contract.panel_title_source == "generic":
                titles.append(f"{contract.panel_title_generic_prefix} {idx+1}")
            else:
                # fallback order
                titles.append(iframe_title or f"{contract.panel_title_generic_prefix} {idx+1}")

        first = group[0]
        acc = _make_dp_accordion(soup, group, titles)
        first.insert_before(acc)
        created += 1
        wrapped += len(group)

        # spacer like your example
        spacer = soup.new_tag("p")
        spacer.string = "\xa0"
        acc.insert_after(spacer)

    return str(root), {"wrapped_iframes": wrapped, "accordions_created": created}


# =============================================================================
# DETERMINISTIC WRAPPER SCAFFOLD
# =============================================================================

def build_scaffold_html(contract: StyleContract, header_title: str, item_title: str) -> str:
    """
    Deterministic scaffold controlled by StyleContract.
    Inject LOCKED_TOKEN later.
    """
    data_title = html_lib.escape(item_title or header_title or "Content")
    category = html_lib.escape(contract.data_category or "Instructional")

    wrapper_classes = contract.wrapper_variant
    header_classes = contract.header_variant

    header_html = ""
    if contract.header_enabled:
        # Keep header simple + stable; you can expand later if you want dp-header-pre spans.
        safe_title = html_lib.escape(header_title or item_title or "")
        header_html = f"""
        <header class="{header_classes}">
          <h2 class="dp-heading">{safe_title}</h2>
        </header>
        """.strip()

    scaffold = f"""
    <div id="dp-wrapper" class="{wrapper_classes}">
      {header_html}
      <div class="dp-content-block" data-category="{category}" data-title="{data_title}">
        {LOCKED_TOKEN}
      </div>
    </div>
    """.strip()

    return scaffold


def apply_scaffold(scaffold: str, locked_html: str) -> str:
    return scaffold.replace(LOCKED_TOKEN, locked_html or "")


# =============================================================================
# LLM: PLANNER + MICRO TASKS (JSON ONLY)
# =============================================================================

def llm_json(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """
    Ask for JSON; parse robustly.
    """
    resp = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()

    # Try direct parse
    try:
        return json.loads(content)
    except Exception:
        # Try extract JSON block
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"_parse_error": True, "_raw": content}


def prompt_header_suggestion(item_title: str, content_preview: str) -> str:
    return textwrap.dedent(f"""
    You suggest a concise, helpful page header title for a Canvas page.

    Rules:
    - Return JSON only.
    - Do NOT invent course numbers or module numbers unless clearly present in the input.
    - Keep it short (<= 60 characters).
    - Prefer using the Canvas item title if it is already good.

    Input:
    - Canvas item title: {item_title}
    - Content preview: {content_preview}

    Output JSON schema:
    {{
      "header_title": "..."
    }}
    """).strip()


def prompt_panel_titles_from_context(item_title: str, html_with_iframes: str) -> str:
    """
    Ask the LLM to label each iframe panel based on nearby visible text.
    Returns mapping by iframe src.
    """
    # Keep prompt small: extract only iframe src + nearby text snippets deterministically.
    soup = BeautifulSoup(html_with_iframes or "", "html.parser")
    root = soup.body if soup.body else soup

    blocks = []
    for iframe in root.find_all("iframe"):
        src = (iframe.get("src") or "").strip()
        title = (iframe.get("title") or "").strip()

        # gather a little preceding text
        prev_texts = []
        # walk backward siblings
        sib = iframe.parent
        if sib and isinstance(sib, Tag):
            # If iframe is inside <p>, check previous siblings of that <p>
            anchor = sib if sib.name in {"p", "div"} else sib
            prev = anchor.previous_sibling
            while prev and len(prev_texts) < 3:
                if isinstance(prev, Tag):
                    t = norm_text(prev.get_text(" ", strip=True))
                    if t:
                        prev_texts.append(t)
                prev = prev.previous_sibling

        blocks.append({
            "src": src,
            "iframe_title": title,
            "preceding_text": list(reversed(prev_texts)),
        })

    return textwrap.dedent(f"""
    You label embedded iframes for accordion panel headings.

    Rules:
    - Return JSON only.
    - Do NOT change URLs.
    - Prefer a short, human-friendly label inferred from preceding text.
    - If nothing useful exists, fall back to iframe_title.
    - Max 60 characters per label.

    Canvas item title: {item_title}

    Iframes (with context):
    {json.dumps(blocks, ensure_ascii=False)}

    Output JSON schema:
    {{
      "panel_titles_by_src": {{
        "<iframe src>": "<panel title>",
        "...": "..."
      }}
    }}
    """).strip()


def prompt_accessibility_review(item_title: str, html_snippet: str) -> str:
    """
    Optional: LLM reviews and returns suggested issues; does not auto-edit.
    """
    return textwrap.dedent(f"""
    You are reviewing Canvas HTML for accessibility issues.

    Rules:
    - Return JSON only.
    - Do not rewrite the HTML.
    - Only report issues that are reasonably inferable from the HTML.
    - Be concise.

    Canvas item title: {item_title}

    HTML:
    {html_snippet}

    Output JSON schema:
    {{
      "issues": [
        {{
          "severity": "low|medium|high",
          "issue": "...",
          "suggestion": "..."
        }}
      ]
    }}
    """).strip()


# =============================================================================
# VALIDATION
# =============================================================================

def extract_attr_set(html: str, tag: str, attr: str) -> set[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    vals = set()
    for t in soup.find_all(tag):
        v = t.get(attr)
        if v:
            vals.add(str(v).strip())
    return vals


def find_legacy_hits(html: str) -> List[str]:
    hits = []
    for pat in LEGACY_PATTERNS:
        if re.search(pat, html or "", flags=re.IGNORECASE):
            hits.append(pat)
    return hits


def original_text_preserved(original_html: str, new_html: str) -> bool:
    """
    With our pipeline, the original HTML is injected unchanged, so original text should be contained.
    Allow extra wrapper/header/panel heading text.
    """
    o = norm_text(visible_text(original_html))
    n = norm_text(visible_text(new_html))
    if not o:
        return True
    return o in n


def validate_item(original_html: str, new_html: str, contract: StyleContract) -> Dict[str, Any]:
    ok_text = original_text_preserved(original_html, new_html)

    o_hrefs = extract_attr_set(original_html, "a", "href")
    n_hrefs = extract_attr_set(new_html, "a", "href")
    o_srcs = extract_attr_set(original_html, "img", "src") | extract_attr_set(original_html, "source", "src")
    n_srcs = extract_attr_set(new_html, "img", "src") | extract_attr_set(new_html, "source", "src")
    o_iframes = extract_attr_set(original_html, "iframe", "src")
    n_iframes = extract_attr_set(new_html, "iframe", "src")

    missing_hrefs = sorted(list(o_hrefs - n_hrefs))
    missing_srcs = sorted(list(o_srcs - n_srcs))
    missing_iframes = sorted(list(o_iframes - n_iframes))
    ok_assets = (not missing_hrefs and not missing_srcs and not missing_iframes)

    legacy_hits = find_legacy_hits(new_html) if contract.remove_legacy else []
    ok_legacy = (len(legacy_hits) == 0)

    ok_iframes_wrapped = True
    if contract.wrap_iframes:
        soup = BeautifulSoup(new_html or "", "html.parser")
        root = soup.body if soup.body else soup
        for iframe in root.find_all("iframe"):
            if not _is_inside_dp_accordion(iframe):
                ok_iframes_wrapped = False
                break

    return {
        "ok": bool(ok_text and ok_assets and ok_legacy and ok_iframes_wrapped),
        "ok_text_preserved": ok_text,
        "ok_assets": ok_assets,
        "ok_no_legacy": ok_legacy,
        "ok_iframes_wrapped": ok_iframes_wrapped,
        "missing_hrefs": missing_hrefs[:30],
        "missing_srcs": missing_srcs[:30],
        "missing_iframes": missing_iframes[:30],
        "legacy_hits": legacy_hits,
    }


# =============================================================================
# MAIN PIPELINE PER ITEM
# =============================================================================

def rewrite_item_hybrid(
    client: OpenAI,
    item: Dict[str, Any],
    contract: StyleContract,
    use_llm_header: bool,
    use_llm_panel_titles: bool,
    run_accessibility_review: bool,
) -> Tuple[str, Dict[str, Any]]:
    original_html = (item.get("original_html") or "").strip()
    item_title = (item.get("title") or "").strip()

    # --- LLM decision: header title (optional, JSON) ---
    header_title = item_title
    header_meta = {}
    if contract.header_enabled:
        if contract.header_title_source == "custom":
            header_title = (contract.header_title_custom or item_title).strip()
        elif contract.header_title_source == "llm_suggest" and use_llm_header:
            data = llm_json(client, prompt_header_suggestion(item_title, preview_text(original_html)))
            header_title = (data.get("header_title") or item_title).strip()
            header_meta = data
        else:
            header_title = item_title

    # --- Deterministic scaffold ---
    scaffold = build_scaffold_html(contract, header_title=header_title, item_title=item_title)
    combined = apply_scaffold(scaffold, original_html)

    # --- Deterministic cleanup ---
    cleanup_info = {}
    if contract.remove_legacy:
        combined, cleanup_info = strip_kl_and_legacy(combined)

    # --- LLM decision: panel titles (optional, JSON) ---
    panel_titles_by_src = None
    panel_meta = {}
    if contract.wrap_iframes and contract.panel_title_source == "preceding_text_llm" and use_llm_panel_titles:
        data = llm_json(client, prompt_panel_titles_from_context(item_title, combined))
        panel_titles_by_src = data.get("panel_titles_by_src") if isinstance(data.get("panel_titles_by_src"), dict) else {}
        panel_meta = data

    # --- Deterministic iframe->DP accordion wrapping ---
    combined, wrap_info = enforce_iframe_accordions(combined, contract, llm_panel_titles=panel_titles_by_src)

    # --- Optional LLM: accessibility review (report only) ---
    a11y = {}
    if run_accessibility_review:
        # keep review bounded: send the combined wrapper but not gigantic pages
        snippet = combined
        if len(snippet) > 12000:
            snippet = snippet[:12000] + "\n<!-- truncated for review -->"
        a11y = llm_json(client, prompt_accessibility_review(item_title, snippet))

    # --- Validate ---
    validation = validate_item(original_html, combined, contract)

    report = {
        "contract": asdict(contract),
        "header_meta": header_meta,
        "panel_meta": panel_meta,
        "cleanup": cleanup_info,
        "wrap": wrap_info,
        "a11y": a11y,
        "validation": validation,
    }
    return combined, report


# =============================================================================
# STREAMLIT STATE
# =============================================================================

for k, v in {
    "content_items": [],
    "course_id": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Canvas Course Rewriter (Hybrid)", layout="wide")
st.title("Canvas Course Rewriter — Hybrid (Deterministic + LLM Micro-Decisions)")

# Sidebar: load target course
st.sidebar.header("Canvas connection")
target_course_id = st.sidebar.text_input("Target course ID", help="Numeric Canvas course ID.")

if st.sidebar.button("Fetch course content"):
    if not target_course_id:
        st.sidebar.error("Enter a course ID.")
    else:
        base_url, token = get_canvas_config()
        try:
            with st.spinner("Fetching pages, assignments, and discussions…"):
                _ = get_course(base_url, token, target_course_id)
                pages = get_pages(base_url, token, target_course_id)
                assignments = get_assignments(base_url, token, target_course_id)
                discussions = get_discussions(base_url, token, target_course_id)

                items: List[Dict[str, Any]] = []

                for p in pages:
                    items.append({
                        "type": "page",
                        "canvas_id": p["page_id"],
                        "url_slug": p["url"],
                        "title": p["title"],
                        "original_html": p.get("body", "") or "",
                        "new_html": "",
                        "approved": False,
                        "selected_for_rewrite": True,
                        "report": {},
                    })

                for a in assignments:
                    items.append({
                        "type": "assignment",
                        "canvas_id": a["id"],
                        "title": a["name"],
                        "original_html": a.get("description", "") or "",
                        "new_html": "",
                        "approved": False,
                        "selected_for_rewrite": True,
                        "report": {},
                    })

                for d in discussions:
                    items.append({
                        "type": "discussion",
                        "canvas_id": d["id"],
                        "title": d["title"],
                        "original_html": d.get("message", "") or "",
                        "new_html": "",
                        "approved": False,
                        "selected_for_rewrite": True,
                        "report": {},
                    })

                st.session_state["content_items"] = items
                st.session_state["course_id"] = target_course_id

            st.success(f"Loaded {len(items)} items.")
        except Exception as e:
            st.sidebar.error(f"Error fetching content: {e}")

# -------------------------------------------------------------------------
# Step: Style Contract UI
# -------------------------------------------------------------------------

st.header("Step 1 — Choose deterministic style settings")

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    wrapper_label = st.selectbox("Wrapper theme preset", list(WRAPPER_PRESETS.keys()), index=0)
    wrapper_variant = WRAPPER_PRESETS[wrapper_label]

with col2:
    header_enabled = st.checkbox("Include DesignPLUS header", value=True)
    header_label = st.selectbox("Header style", list(HEADER_PRESETS.keys()), index=0)
    header_variant = HEADER_PRESETS[header_label]

with col3:
    data_category = st.selectbox("data-category", ["Instructional", "Overview", "Assessment", "Resources"], index=0)

st.subheader("Header title policy")

hcol1, hcol2 = st.columns([2, 3])
with hcol1:
    header_title_source = st.selectbox(
        "Header title source",
        ["canvas_title", "llm_suggest", "custom"],
        index=0,
        help="LLM suggest = model proposes a better header title (JSON).",
    )
with hcol2:
    header_title_custom = st.text_input("Custom header title (if selected)", value="")

st.subheader("Accordion / iframe policy")

acol1, acol2, acol3 = st.columns([2, 2, 2])
with acol1:
    wrap_iframes = st.checkbox("Wrap iframes into canonical DP accordions", value=True)
with acol2:
    iframe_grouping = st.selectbox("Iframe grouping", ["group_consecutive", "one_per_iframe"], index=0)
with acol3:
    panel_title_source = st.selectbox(
        "Accordion panel title source",
        ["iframe_title", "preceding_text_llm", "generic"],
        index=0,
        help="preceding_text_llm = model labels panels using nearby text (JSON).",
    )

panel_title_generic_prefix = st.text_input("Generic panel prefix", value="Panel")

remove_legacy = st.checkbox("Remove legacy / kl_* markup if present", value=True)

contract = StyleContract(
    wrapper_variant=wrapper_variant,
    header_enabled=header_enabled,
    header_variant=header_variant,
    header_title_source=header_title_source,
    header_title_custom=header_title_custom,
    data_category=data_category,
    wrap_iframes=wrap_iframes,
    iframe_grouping=iframe_grouping,
    panel_title_source=panel_title_source,
    panel_title_generic_prefix=panel_title_generic_prefix,
    remove_legacy=remove_legacy,
)

with st.expander("View Style Contract JSON", expanded=False):
    st.code(json.dumps(asdict(contract), indent=2), language="json")

# -------------------------------------------------------------------------
# Step: LLM micro-decision toggles
# -------------------------------------------------------------------------

st.header("Step 2 — Choose which decision tasks to delegate to the model")

dcol1, dcol2, dcol3 = st.columns([2, 2, 2])
with dcol1:
    use_llm_header = st.checkbox("LLM: suggest header title", value=(header_title_source == "llm_suggest"))
with dcol2:
    use_llm_panel_titles = st.checkbox("LLM: label accordion panels from nearby text", value=(panel_title_source == "preceding_text_llm"))
with dcol3:
    run_accessibility_review = st.checkbox("LLM: accessibility review report (no auto-fixes)", value=False)

st.caption("The model never rewrites your content HTML. It only returns JSON decisions.")

# -------------------------------------------------------------------------
# Step: Select items to rewrite
# -------------------------------------------------------------------------

st.header("Step 3 — Select items to process")

items = st.session_state["content_items"]

if not items:
    st.info("Load course content from the sidebar.")
else:
    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    with f1:
        show_pages = st.checkbox("Pages", value=True)
    with f2:
        show_assignments = st.checkbox("Assignments", value=True)
    with f3:
        show_discussions = st.checkbox("Discussions", value=True)
    with f4:
        search = st.text_input("Search titles", value="")

    a1, a2, _ = st.columns([1, 1, 3])
    with a1:
        if st.button("Select all shown"):
            for it in items:
                if it["type"] == "page" and not show_pages:
                    continue
                if it["type"] == "assignment" and not show_assignments:
                    continue
                if it["type"] == "discussion" and not show_discussions:
                    continue
                if search and search.lower() not in it["title"].lower():
                    continue
                it["selected_for_rewrite"] = True
    with a2:
        if st.button("Select none shown"):
            for it in items:
                if it["type"] == "page" and not show_pages:
                    continue
                if it["type"] == "assignment" and not show_assignments:
                    continue
                if it["type"] == "discussion" and not show_discussions:
                    continue
                if search and search.lower() not in it["title"].lower():
                    continue
                it["selected_for_rewrite"] = False

    shown = 0
    for idx, it in enumerate(items):
        if it["type"] == "page" and not show_pages:
            continue
        if it["type"] == "assignment" and not show_assignments:
            continue
        if it["type"] == "discussion" and not show_discussions:
            continue
        if search and search.lower() not in it["title"].lower():
            continue
        shown += 1
        it["selected_for_rewrite"] = st.checkbox(
            f"[{it['type']}] {it['title']}",
            value=bool(it.get("selected_for_rewrite", True)),
            key=f"sel_{idx}",
        )

    st.caption(f"Showing {shown} items.")

# -------------------------------------------------------------------------
# Step: Run
# -------------------------------------------------------------------------

st.header("Step 4 — Run processing")

can_run = bool(items)
if st.button("Run on selected items", disabled=not can_run):
    client = get_ai_client()
    selected = [it for it in items if it.get("selected_for_rewrite", True)]
    if not selected:
        st.warning("No items selected.")
        st.stop()

    prog = st.progress(0.0)
    status = st.empty()

    for i, it in enumerate(selected):
        status.write(f"Processing [{it['type']}] {it['title']}…")
        try:
            new_html, report = rewrite_item_hybrid(
                client=client,
                item=it,
                contract=contract,
                use_llm_header=use_llm_header,
                use_llm_panel_titles=use_llm_panel_titles,
                run_accessibility_review=run_accessibility_review,
            )
            it["new_html"] = new_html
            it["report"] = report
            it.pop("error", None)
        except Exception as e:
            it["error"] = str(e)
            it["report"] = {"validation": {"ok": False}, "error": str(e)}
        prog.progress((i + 1) / len(selected))

    status.write("Done.")

# -------------------------------------------------------------------------
# Review / Approve
# -------------------------------------------------------------------------

st.header("Step 5 — Review, approve, and write back")

show_details = st.checkbox("Show detailed report per item", value=False)

if items:
    for idx, it in enumerate(items):
        with st.expander(f"[{it['type']}] {it['title']}", expanded=False):
            if it.get("error"):
                st.error(it["error"])

            report = it.get("report") or {}
            validation = (report.get("validation") or {})
            if validation:
                if validation.get("ok"):
                    st.success("Validation OK (text preserved + assets preserved + legacy banned + iframes wrapped).")
                else:
                    st.warning("Validation FAILED")
                    if validation.get("ok_text_preserved") is False:
                        st.write("- Original visible text not preserved (unexpected; check injection).")
                    if validation.get("missing_iframes"):
                        st.write(f"- Missing iframe(s): {validation['missing_iframes'][:5]}")
                    if validation.get("missing_hrefs"):
                        st.write(f"- Missing href(s): {validation['missing_hrefs'][:5]}")
                    if validation.get("missing_srcs"):
                        st.write(f"- Missing src(s): {validation['missing_srcs'][:5]}")
                    if validation.get("legacy_hits"):
                        st.write(f"- Legacy hits: {validation['legacy_hits']}")
                    if validation.get("ok_iframes_wrapped") is False:
                        st.write("- Some iframes are not inside dp-panels-wrapper accordions")

            colA, colB = st.columns(2)
            with colA:
                st.subheader("Original (visual)")
                components.html(it.get("original_html") or "", height=320, scrolling=True)
            with colB:
                st.subheader("Proposed (visual)")
                if it.get("new_html"):
                    components.html(it["new_html"], height=320, scrolling=True)
                else:
                    st.info("Not processed yet.")

            it["approved"] = st.checkbox("Approve", value=bool(it.get("approved", False)), key=f"appr_{idx}")

            if show_details and report:
                st.subheader("Report")
                st.code(json.dumps(report, indent=2), language="json")

                a11y = report.get("a11y") or {}
                issues = a11y.get("issues") if isinstance(a11y, dict) else None
                if issues:
                    st.subheader("Accessibility review")
                    for iss in issues[:20]:
                        st.write(f"- **{iss.get('severity','low')}**: {iss.get('issue','')}")
                        if iss.get("suggestion"):
                            st.caption(iss["suggestion"])

# -------------------------------------------------------------------------
# Write back
# -------------------------------------------------------------------------

if st.button("Write approved changes to Canvas"):
    if not st.session_state.get("course_id"):
        st.error("No course loaded.")
    else:
        base_url, token = get_canvas_config()
        course_id = st.session_state["course_id"]

        approved = [it for it in items if it.get("approved") and it.get("new_html")]
        if not approved:
            st.warning("No approved items to write.")
        else:
            with st.spinner(f"Writing {len(approved)} item(s) to Canvas…"):
                errs = []
                for it in approved:
                    try:
                        if it["type"] == "page":
                            update_page_html(base_url, token, course_id, it["url_slug"], it["new_html"])
                        elif it["type"] == "assignment":
                            update_assignment_html(base_url, token, course_id, it["canvas_id"], it["new_html"])
                        elif it["type"] == "discussion":
                            update_discussion_html(base_url, token, course_id, it["canvas_id"], it["new_html"])
                    except Exception as e:
                        errs.append((it["title"], str(e)))

            if errs:
                st.error("Some updates failed:")
                for title, msg in errs:
                    st.write(f"- **{title}**: {msg}")
            else:
                st.success("All approved items were written to Canvas.")
