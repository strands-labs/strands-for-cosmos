"""CI gate for the docs site: the three ways it broke on 2026-09-30.

1. A raw-HTML asset path that starts with ``/`` is pinned to a site root. The
   site lives at ``/strands-for-cosmos/`` (site_url), and ``/strands-cosmos/…``
   404'd on every page that used it. Raw ``<img>``/``<source>``/``<a>`` paths in
   the markdown must be relative; mkdocs only rewrites markdown-syntax links.
2. A mermaid fence with a malformed edge (``A --> RCosmos 3``) is a parse error
   at render time. Mermaid then leaves its error box in ``<body>`` and, with
   ``navigation.instant``, that box follows the reader onto every other page.
   Every fence is checked for balanced brackets/quotes and well-formed edge
   targets; nothing here needs node.
3. The header shows top-level nav titles inline on one row (overrides/partials/
   header.html). A title with whitespace wraps to two lines at the 76.25em
   breakpoint, so top-level titles are single words.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MD_FILES = sorted(p for p in DOCS.rglob("*.md"))

_ATTR_PATH = re.compile(r"""\b(?:src|href)\s*=\s*["'](/[^"']*)["']""")
_FENCE = re.compile(r"^```mermaid[^\n]*\n(.*?)^```", re.S | re.M)


def test_docs_have_markdown_pages():
    assert MD_FILES, "docs/ has no markdown pages"


def test_no_site_root_absolute_asset_paths():
    offenders = []
    for md in MD_FILES:
        for lineno, line in enumerate(md.read_text().splitlines(), 1):
            for path in _ATTR_PATH.findall(line):
                if path.startswith("//"):
                    continue  # protocol-relative external URL
                offenders.append(f"{md.relative_to(ROOT)}:{lineno}: {path}")
    assert not offenders, (
        "site-root-absolute asset paths break under the /strands-for-cosmos/ prefix; "
        "use a path relative to the page URL:\n  " + "\n  ".join(offenders)
    )


def _mermaid_fences():
    for md in MD_FILES:
        text = md.read_text()
        for m in _FENCE.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            yield md, lineno, m.group(1)


_ARROW = re.compile(r"(?:<?-{2,3}>?|-\.+->|={2,3}>?|--\s[^>|]*?\s-->)")
_LABEL = re.compile(r"^\s*\|[^|]*\|")
# A flowchart node reference: an id, optionally followed by a shape opener.
_NODE = re.compile(r"""^\s*[\w.-]+\s*(?:$|;|\[|\(|\{|>|:::|&)""")


def _check_fence(body: str) -> list[str]:
    problems = []
    first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    if body.count('"') % 2:
        problems.append("odd number of double quotes")
    stripped = re.sub(r'"[^"]*"', '""', body)
    for open_, close in ("[]", "()", "{}"):
        if stripped.count(open_) != stripped.count(close):
            problems.append(f"unbalanced {open_}{close}")
    if first.startswith(("graph", "flowchart")):
        for ln in stripped.splitlines():
            code = ln.split("%%", 1)[0]
            if not code.strip() or code.strip().startswith(("style", "classDef", "class ", "subgraph", "end", "linkStyle", "click")):
                continue
            parts = _ARROW.split(code)
            for target in parts[1:]:
                target = _LABEL.sub("", target)
                if not target.strip():
                    continue
                if not _NODE.match(target):
                    problems.append(f"malformed edge target: {target.strip()!r}")
    return problems


def test_mermaid_fences_are_well_formed():
    offenders = []
    for md, lineno, body in _mermaid_fences():
        for p in _check_fence(body):
            offenders.append(f"{md.relative_to(ROOT)}:{lineno}: {p}")
    assert not offenders, "mermaid fences that will not render:\n  " + "\n  ".join(offenders)


def test_mermaid_grader_catches_the_home_page_bug():
    bad = 'graph LR\n    A["Agent"] --> RCosmos 3\n    R -->|Reasoner| U["Understand"]\n'
    assert any("malformed edge target" in p for p in _check_fence(bad))
    good = 'graph LR\n    A["Agent"] --> R["Cosmos 3"]\n    R -->|Reasoner| U["Understand"]\n    R --> V\n'
    assert _check_fence(good) == []


def test_top_level_nav_titles_fit_one_header_row():
    class Loose(yaml.SafeLoader):
        pass

    Loose.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix)
    cfg = yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=Loose)
    titles = [next(iter(item)) for item in cfg["nav"] if isinstance(item, dict)]
    wide = [t for t in titles if re.search(r"\s", t) or len(t) > 12]
    assert not wide, f"top-level nav titles wrap in the one-row header: {wide}"
