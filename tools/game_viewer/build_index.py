#!/usr/bin/env python3
"""Sync blog/index.md -> blog/index.html (the Markdown is the source of truth).

The author edits `blog/index.md`; this regenerates a styled, self-contained HTML
rendering so the prose can be read as a page and the widgets shown in place.
Run it after every edit to the Markdown:

    python -m tools.game_viewer.build_index                 # blog/index.md -> blog/index.html
    python -m tools.game_viewer.build_index --md P --out Q

Gameplay viewers (e7_viewer.html, e8*_viewer.html, all_games.html) are separate
standalone files in the same blog/ directory; references to them in the prose
(`e7_viewer.html` in backticks) become links. Raw HTML blocks in the Markdown
pass through untouched.

In-article gameplay embeds: where the Markdown carries

    <div class="game-embed" data-demo="opening"></div>

this splices in `blog/embeds/opening.html` -- a self-contained widget built by
`tools/game_viewer/build_demo.py`. The marker renders as nothing in a plain
Markdown previewer, so the source stays readable; the built page shows the game.

Table convention: a table's final row is treated as a summary/total row (and
given a heavier rule above it) when its cells are bolded -- e.g.
`| **Mean (seeds 0-4)** | **3.06** | ... |`.
"""
from __future__ import annotations
import argparse
import os
import re

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_MD = os.path.join(ROOT, "blog", "index.md")

CSS = """
:root{
  --bg:#fbfaf7; --surface:#ffffff; --ink:#23202a; --ink2:#544c5e; --muted:#8a8194;
  --accent:#7a3b8f; --accent2:#b8622a; --line:#e7e2df; --code-bg:#f2eef4;
  --th-bg:#f4eff6; --row:#faf8fb;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#15131a; --surface:#1c1a22; --ink:#ece8f0; --ink2:#c3bccd; --muted:#8a8194;
    --accent:#c88fda; --accent2:#e0975c; --line:#2c2833; --code-bg:#241f2c;
    --th-bg:#241f2c; --row:#1f1c27;
  }
}
:root[data-theme="light"]{
  --bg:#fbfaf7; --surface:#ffffff; --ink:#23202a; --ink2:#544c5e; --muted:#8a8194;
  --accent:#7a3b8f; --accent2:#b8622a; --line:#e7e2df; --code-bg:#f2eef4;
  --th-bg:#f4eff6; --row:#faf8fb;
}
:root[data-theme="dark"]{
  --bg:#15131a; --surface:#1c1a22; --ink:#ece8f0; --ink2:#c3bccd; --muted:#8a8194;
  --accent:#c88fda; --accent2:#e0975c; --line:#2c2833; --code-bg:#241f2c;
  --th-bg:#241f2c; --row:#1f1c27;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16.5px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;}
main{max-width:44rem;margin:0 auto;padding:3.2rem 1.4rem 6rem;}
h1,h2,h3,h4{line-height:1.2;text-wrap:balance;color:var(--ink);margin:2.4em 0 .6em;font-weight:650;}
h1{font-size:2.15rem;margin-top:.2em;letter-spacing:-.01em;}
h1+*{margin-top:.4em}
h2{font-size:1.5rem;padding-bottom:.28em;border-bottom:1px solid var(--line);}
h3{font-size:1.2rem;color:var(--accent);}
p,li{color:var(--ink);}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid color-mix(in oklab,var(--accent) 40%,transparent);}
a:hover{border-bottom-color:var(--accent);}
strong{font-weight:650;color:var(--ink);}
em{color:var(--ink2);}
code{background:var(--code-bg);border-radius:4px;padding:.08em .34em;font-size:.9em;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
pre{background:var(--code-bg);border-radius:8px;padding:1em;overflow-x:auto;}
pre code{background:none;padding:0;}
blockquote{margin:1.2em 0;padding:.2em 1.1em;border-left:3px solid var(--accent2);color:var(--ink2);}
hr{border:0;border-top:1px solid var(--line);margin:2.4em 0;}
sup a{border:0;font-size:.8em;}
.tablewrap{overflow-x:auto;margin:1.4em 0;}
table{border-collapse:collapse;width:100%;font-size:.92rem;
  font-variant-numeric:tabular-nums;background:var(--surface);
  border:1px solid var(--line);border-radius:8px;overflow:hidden;}
th,td{text-align:left;padding:.5em .72em;border-bottom:1px solid var(--line);vertical-align:top;}
th{background:var(--th-bg);font-weight:600;color:var(--ink);}
tbody tr:nth-child(even){background:var(--row);}
tbody tr:last-child td{border-bottom:0;}
/* a bolded final row is a summary/total row: set it off with a heavier rule */
tbody tr:last-child:has(strong) td{border-top:2px solid var(--ink2);}
td code,th code{background:color-mix(in oklab,var(--code-bg) 60%,transparent);}
ol,ul{padding-left:1.4em;}
li{margin:.3em 0;}
img{max-width:100%;height:auto;}
.masthead{color:var(--muted);font-size:.82rem;letter-spacing:.04em;text-transform:uppercase;
  margin-bottom:2.2rem;}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main>
<p class="masthead">NetHack · agent evaluation · working draft</p>
{body}
</main>
</body>
</html>
"""


def render_md(text: str) -> str:
    md = markdown.Markdown(extensions=[
        "tables", "fenced_code", "footnotes", "sane_lists", "attr_list", "md_in_html",
    ])
    html = md.convert(text)
    # Wrap tables so wide ones scroll inside their own box, never the page body.
    html = html.replace("<table>", '<div class="tablewrap"><table>').replace(
        "</table>", "</table></div>")
    # Make references to the standalone gameplay viewers clickable.
    html = re.sub(r"<code>([\w./-]+\.html)</code>",
                  r'<a href="\1"><code>\1</code></a>', html)
    return html


EMBED_RE = re.compile(
    r'<div class="game-embed" data-demo="([\w-]+)"\s*>\s*</div>')


def splice_embeds(html: str, embed_dir: str) -> str:
    """Replace `<div class="game-embed" data-demo="X">` with blog/embeds/X.html."""
    def sub(m):
        path = os.path.join(embed_dir, m.group(1) + ".html")
        if not os.path.exists(path):
            raise SystemExit(f"missing embed {path} -- run tools.game_viewer.build_demo")
        return open(path, encoding="utf-8").read()
    return EMBED_RE.sub(sub, html)


def build(md_path: str, out_path: str) -> str:
    text = open(md_path, encoding="utf-8").read()
    body = render_md(text)
    body = splice_embeds(body, os.path.join(os.path.dirname(os.path.abspath(md_path)), "embeds"))
    m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    title = (m.group(1).strip().rstrip(".") if m else "NetHack blog")
    html = TEMPLATE.format(title=title, css=CSS, body=body)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sync blog/index.md -> blog/index.html")
    ap.add_argument("--md", default=DEFAULT_MD)
    ap.add_argument("--out", default=None, help="default: index.html beside the .md")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.md)), "index.html")
    path = build(args.md, out)
    print(f"wrote {path} ({os.path.getsize(path)/1024:.0f} KB) from {args.md}")


if __name__ == "__main__":
    main()
