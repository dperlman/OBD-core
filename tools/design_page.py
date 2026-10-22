#!/usr/bin/env python3
"""The page "OBD Data Store Design", generated from docs/DESIGN.md.

    python ~/git/OBD-core/tools/design_page.py OUT.html [--repo ~/git/OBD-core]

docs/DESIGN.md is the design document; the page (claude.ai/artifact/FgtACNaNpxvsap5rSqFyS7) is a
copy for easier reading and is never edited by hand.  After DESIGN.md changes, commit it, run this,
and publish OUT.html to that page.  The footer names the DESIGN.md commit the page was made from,
so commit first: with uncommitted changes the footer says so.

Needs the `markdown` package, which obd-core itself does not install (the research repo's .venv
has it).  Not part of the package; nothing imports it.
"""
import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.exit("design_page.py needs the markdown package: pip install markdown "
             "(~/git/ordered-binomial-cusps/.venv has it)")

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("out", type=Path, help="the HTML file to write")
ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent,
                help="the OBD-core checkout (default: the one this script is in)")
args = ap.parse_args()
repo, out = args.repo.expanduser(), args.out


def git(*a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                          check=True).stdout.strip()


src = (repo / "docs" / "DESIGN.md").read_text()
sha, date = git("log", "-1", "--format=%h %ad", "--date=short", "--", "docs/DESIGN.md").split()
if git("status", "--porcelain", "--", "docs/DESIGN.md"):
    print("warning: docs/DESIGN.md has uncommitted changes; commit it first", file=sys.stderr)
    sha += " plus uncommitted changes"

md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"],
                       extension_configs={"toc": {"toc_depth": "2-2"}})
body = md.convert(src)

# The first heading and first paragraph become the page header.
m = re.match(r'\s*<h1[^>]*>(.*?)</h1>\s*<p>(.*?)</p>', body, re.S)
lede, body = m.group(2), body[m.end():]
body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
body = body.replace("<pre>", '<pre class="code">')
toc = "\n".join(f'      <li><a href="#{t["id"]}">{t["name"]}</a></li>' for t in md.toc_tokens)

page = f"""<title>OBD Data Store Design</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,650&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column (~72ch) with a sticky contents rail on wide screens; tables and code scroll in their own boxes. */
:root {{
  --bg: #f7f8fa; --surface: #ffffff; --fg: #1b2230; --muted: #5a6475; --rule: #d9dee6;
  --fall: #1f5a9e;          /* the plots' blue: E falling */
  --cusp: #f2d43d;          /* the plots' bright cusp yellow, a highlight only */
  --code-bg: #eef1f5; --cusp-ink: #3d3300;
  --display: "Literata", Georgia, "Times New Roman", serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #11151c; --surface: #171c25; --fg: #e3e7ee; --muted: #9aa4b4; --rule: #2a3240;
    --fall: #7fb0ea; --cusp: #f2d43d; --code-bg: #1d2430; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #11151c; --surface: #171c25; --fg: #e3e7ee; --muted: #9aa4b4; --rule: #2a3240;
  --fall: #7fb0ea; --cusp: #f2d43d; --code-bg: #1d2430; color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font: 400 1.0625rem/1.6 var(--body); padding-inline: 20px; padding-block: 0 4rem; }}
.wrap {{ max-width: 1120px; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 1fr); gap: 2.5rem; }}
@media (min-width: 980px) {{ .wrap {{ grid-template-columns: 200px minmax(0, 1fr); }} }}
nav.toc {{ display: none; }}
@media (min-width: 980px) {{
  nav.toc {{ display: block; position: sticky; top: calc(env(safe-area-inset-top, 0px) + 1.5rem); align-self: start; margin-top: 7.5rem; font-size: .875rem; }}
  nav.toc ol {{ list-style: none; padding: 0; margin: 0; display: grid; gap: .45rem; }}
  nav.toc a {{ color: var(--muted); text-decoration: none; }}
  nav.toc a:hover, nav.toc a:focus-visible {{ color: var(--fg); text-decoration: underline; }}
}}
main {{ min-width: 0; max-width: 72ch; }}
header {{ padding-block: 3rem 1.5rem; border-bottom: 1px solid var(--rule); margin-bottom: 1.5rem; }}
.eyebrow {{ font: 500 .75rem/1.5 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0 0 .9rem; }}
h1 {{ font: 650 clamp(2rem, 5vw, 2.75rem)/1.1 var(--display); margin: 0 0 .9rem; text-wrap: balance; }}
.lede {{ font-size: 1.15rem; color: var(--muted); margin: 0; }}
h2 {{ font: 650 1.5rem/1.25 var(--display); margin: 3rem 0 .75rem; text-wrap: balance; scroll-margin-top: 1rem; }}
p, ul, ol {{ margin: 0 0 1rem; }}
li {{ margin-bottom: .35rem; }}
a {{ color: var(--fall); overflow-wrap: anywhere; }}
strong {{ font-weight: 600; }}
main > p:first-of-type strong:first-child {{ background: var(--cusp); color: var(--cusp-ink); padding: 0 .25em; border-radius: 3px; }}
code {{ font: 400 .88em/1 var(--mono); background: var(--code-bg); padding: .12em .35em; border-radius: 4px; }}
pre.code {{ font: 400 .84rem/1.55 var(--mono); background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1rem 1.1rem; overflow-x: auto; margin: 0 0 1.2rem; }}
pre.code code {{ background: none; padding: 0; font: inherit; }}
.tbl {{ overflow-x: auto; margin: 0 0 1.3rem; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: .93rem; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: .55rem .8rem; border-bottom: 1px solid var(--rule); vertical-align: top; }}
th {{ font-weight: 600; font-size: .8rem; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); background: var(--code-bg); }}
tr:last-child td {{ border-bottom: 0; }}
footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule); color: var(--muted); font-size: .875rem; }}
:focus-visible {{ outline: 2px solid var(--fall); outline-offset: 2px; }}
</style>

<div class="wrap">
  <nav class="toc" aria-label="Contents">
    <ol>
{toc}
    </ol>
  </nav>
  <main>
    <header>
      <p class="eyebrow">OBD-core · OBD · ordered-binomial-cusps &nbsp;·&nbsp; generated from docs/DESIGN.md, {html.escape(date)}</p>
      <h1>OBD Data Store Design</h1>
      <p class="lede">{lede}</p>
    </header>
{body}
    <footer>Generated from OBD-core <code>docs/DESIGN.md</code> at commit {html.escape(sha)}. That file is the design document: change it there, then regenerate this page. Never edit this page by hand.</footer>
  </main>
</div>
"""
out.write_text(page)
print(out, len(page), "bytes;", len(md.toc_tokens), "sections")
