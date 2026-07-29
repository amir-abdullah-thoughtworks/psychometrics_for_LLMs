#!/usr/bin/env python3
"""
Live preview for the README and docs/ guides.

    python3 docs/preview.py

Serves a rendered view at http://127.0.0.1:8765 and opens it.
The markdown files stay the editable source — edit them in any editor and the
browser re-renders on save. Ctrl-C to stop.

Standard library only.
"""

import http.server
import json
import os
import socketserver
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # repo root
PORT = int(os.environ.get("PREVIEW_PORT", "8765"))

DOCS = [
    ("README.md", "README", "The front page"),
    ("docs/customizing.md", "Customizing", "Personas, instruments, validation"),
    ("docs/repo-map.md", "Repo map", "File-by-file reference"),
    ("DATASETS.md", "Datasets", "HuggingFace dataset inventory"),
]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Psychometrics — docs</title>
<style>
:root {
  --ground:      #f4f5f3;
  --surface:     #ffffff;
  --ink:         #1b2021;
  --ink-soft:    #5a6260;
  --ink-faint:   #8b938f;
  --rule:        #dcdfdb;
  --accent:      #2f5d50;
  --accent-soft: #e3ede8;
  --code-bg:     #eceeea;

  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans:  -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono:  ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground:      #14181a;
    --surface:     #1b2124;
    --ink:         #e6e9e6;
    --ink-soft:    #9aa5a1;
    --ink-faint:   #6d7873;
    --rule:        #2c3438;
    --accent:      #7fbfa6;
    --accent-soft: #1e2e29;
    --code-bg:     #101416;
  }
}
:root[data-theme="dark"] {
  --ground: #14181a; --surface: #1b2124; --ink: #e6e9e6; --ink-soft: #9aa5a1;
  --ink-faint: #6d7873; --rule: #2c3438; --accent: #7fbfa6; --accent-soft: #1e2e29;
  --code-bg: #101416;
}
:root[data-theme="light"] {
  --ground: #f4f5f3; --surface: #ffffff; --ink: #1b2021; --ink-soft: #5a6260;
  --ink-faint: #8b938f; --rule: #dcdfdb; --accent: #2f5d50; --accent-soft: #e3ede8;
  --code-bg: #eceeea;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.65;
  display: grid;
  grid-template-columns: 264px minmax(0, 1fr);
  min-height: 100vh;
}

/* ---------- sidebar ---------- */
aside {
  border-right: 1px solid var(--rule);
  padding: 32px 24px 48px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 28px;
}
.brand { display: flex; flex-direction: column; gap: 6px; }
.brand h1 {
  font-family: var(--serif);
  font-size: 20px;
  line-height: 1.25;
  margin: 0;
  font-weight: 600;
  letter-spacing: -0.01em;
}
.status {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--accent);
  background: var(--accent-soft);
  border-radius: 3px;
  padding: 3px 7px;
  align-self: flex-start;
  font-weight: 600;
}
nav { display: flex; flex-direction: column; gap: 2px; }
nav a.doc {
  display: block;
  padding: 7px 10px;
  border-radius: 4px;
  text-decoration: none;
  color: var(--ink);
  border-left: 2px solid transparent;
}
nav a.doc small { display: block; color: var(--ink-faint); font-size: 12px; line-height: 1.4; }
nav a.doc:hover { background: var(--code-bg); }
nav a.doc[aria-current="true"] {
  background: var(--accent-soft);
  border-left-color: var(--accent);
  color: var(--accent);
  font-weight: 600;
}
nav a.doc[aria-current="true"] small { color: var(--accent); opacity: 0.75; font-weight: 400; }
.sections { display: flex; flex-direction: column; gap: 1px; margin: 6px 0 0 12px; }
.sections a {
  font-size: 13px;
  color: var(--ink-soft);
  text-decoration: none;
  padding: 3px 8px;
  border-left: 1px solid var(--rule);
  line-height: 1.4;
}
.sections a:hover { color: var(--accent); border-left-color: var(--accent); }
.foot { margin-top: auto; font-size: 12px; color: var(--ink-faint); line-height: 1.55; }
.foot code { font-family: var(--mono); font-size: 11px; }

/* ---------- article ---------- */
main { padding: 56px 48px 120px; min-width: 0; }
article { max-width: 74ch; }
article > *:first-child { margin-top: 0; }

h1, h2, h3, h4 {
  font-family: var(--serif);
  font-weight: 600;
  text-wrap: balance;
  letter-spacing: -0.012em;
  line-height: 1.22;
}
h1 { font-size: 38px; margin: 0 0 20px; }
h2 {
  font-size: 25px;
  margin: 52px 0 14px;
  padding-top: 22px;
  border-top: 1px solid var(--rule);
}
h3 { font-size: 19px; margin: 34px 0 10px; }
h4 { font-size: 16px; margin: 26px 0 8px; }
p { margin: 0 0 16px; }
p:has(> img) { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
img { max-width: 100%; vertical-align: middle; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:has(> img) { line-height: 0; }
strong { font-weight: 650; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 40px 0; }

ul, ol { margin: 0 0 16px; padding-left: 22px; display: flex; flex-direction: column; gap: 7px; }
li { line-height: 1.6; }

blockquote {
  margin: 22px 0;
  padding: 14px 18px;
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  border-radius: 0 4px 4px 0;
}
blockquote p:last-child { margin-bottom: 0; }

code {
  font-family: var(--mono);
  font-size: 0.87em;
  background: var(--code-bg);
  padding: 1.5px 5px;
  border-radius: 3px;
  word-break: break-word;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 5px;
  padding: 16px 18px;
  overflow-x: auto;
  margin: 0 0 20px;
  line-height: 1.55;
}
pre code { background: none; padding: 0; font-size: 13px; word-break: normal; }

.tablewrap { overflow-x: auto; margin: 0 0 22px; }
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 14.5px;
  font-variant-numeric: tabular-nums;
}
th, td {
  text-align: left;
  vertical-align: top;
  padding: 9px 14px 9px 0;
  border-bottom: 1px solid var(--rule);
}
th {
  font-size: 11.5px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--ink-soft);
  font-weight: 600;
  border-bottom-color: var(--ink-faint);
}
tbody tr:last-child td { border-bottom: none; }
td code { font-size: 0.85em; }

.toast {
  position: fixed;
  right: 20px;
  bottom: 20px;
  background: var(--accent);
  color: var(--ground);
  padding: 8px 14px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 600;
  opacity: 0;
  transition: opacity 0.25s;
  pointer-events: none;
}
.toast.show { opacity: 1; }

@media (max-width: 900px) {
  body { grid-template-columns: 1fr; }
  aside {
    position: static; height: auto; border-right: none;
    border-bottom: 1px solid var(--rule); padding: 20px;
  }
  .sections { display: none; }
  main { padding: 32px 20px 80px; }
  h1 { font-size: 30px; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }
</style>
</head>
<body>
<aside>
  <div class="brand">
    <h1>LLM Psychometrics</h1>
    <span class="status">Docs preview</span>
  </div>
  <nav id="nav"></nav>
  <div class="foot">
    Live preview. Edit the <code>.md</code> files in any editor — this page
    re-renders on save.
  </div>
</aside>
<main><article id="doc"></article></main>
<div class="toast" id="toast">Reloaded</div>

<script>
const DOCS = __DOCS__;
let current = location.hash.slice(1) || DOCS[0].path;
let stamps = null;

/* ---------- markdown ---------- */
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function inline(text) {
  const code = [];
  let s = text.replace(/`([^`]+)`/g, (_, c) => {
    code.push(c);
    return "\u0000" + (code.length - 1) + "\u0000";
  });
  s = esc(s);
  s = s.replace(/&amp;(nbsp|mdash|ndash|amp|lt|gt|#\d+);/g, "&$1;");   // keep entities
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, '<img src="$2" alt="$1">');
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2">$1</a>');
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => "<code>" + esc(code[+i]) + "</code>");
  return s;
}

const cells = row =>
  row.replace(/^\||\|$/g, "").split("|").map(c => c.trim());

function render(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;

  const isTableSep = l => /^\|[\s:|-]+\|$/.test(l.trim());

  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();

    if (t.startsWith("```")) {                       // fenced code
      const lang = t.slice(3).trim();
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) buf.push(lines[i++]);
      i++;
      out.push('<pre><code class="lang-' + esc(lang) + '">' + esc(buf.join("\n")) + "</code></pre>");
      continue;
    }
    if (!t) { i++; continue; }                       // blank
    if (/^-{3,}$/.test(t) || /^\*{3,}$/.test(t)) { out.push("<hr>"); i++; continue; }

    const h = t.match(/^(#{1,6})\s+(.*)$/);          // heading
    if (h) {
      const lvl = h[1].length;
      const txt = inline(h[2]);
      const id = h[2].toLowerCase().replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-");
      out.push("<h" + lvl + ' id="' + id + '">' + txt + "</h" + lvl + ">");
      i++;
      continue;
    }
    if (t.startsWith(">")) {                         // blockquote
      const buf = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        buf.push(lines[i].trim().replace(/^>\s?/, ""));
        i++;
      }
      out.push("<blockquote><p>" + inline(buf.join(" ")) + "</p></blockquote>");
      continue;
    }
    if (t.startsWith("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = cells(t);                         // table
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(cells(lines[i].trim()));
        i++;
      }
      const blankHead = head.every(c => c === "");
      let html = '<div class="tablewrap"><table>';
      if (!blankHead) {
        html += "<thead><tr>" + head.map(c => "<th>" + inline(c) + "</th>").join("") + "</tr></thead>";
      }
      html += "<tbody>";
      for (const r of rows) {
        html += "<tr>" + r.map(c => "<td>" + inline(c) + "</td>").join("") + "</tr>";
      }
      out.push(html + "</tbody></table></div>");
      continue;
    }
    const ulm = t.match(/^[-*]\s+(.*)$/);            // unordered list
    if (ulm) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim()) && lines[i].trim()) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i++;
      }
      out.push("<ul>" + items.map(x => "<li>" + inline(x) + "</li>").join("") + "</ul>");
      continue;
    }
    const olm = t.match(/^\d+\.\s+(.*)$/);           // ordered list
    if (olm) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim()) && lines[i].trim()) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i++;
      }
      out.push("<ol>" + items.map(x => "<li>" + inline(x) + "</li>").join("") + "</ol>");
      continue;
    }

    const buf = [];                                   // paragraph
    while (i < lines.length) {
      const l = lines[i], lt = l.trim();
      if (!lt || lt.startsWith("```") || lt.startsWith("|") || lt.startsWith(">") ||
          /^#{1,6}\s/.test(lt) || /^[-*]\s+/.test(lt) || /^\d+\.\s+/.test(lt) ||
          /^-{3,}$/.test(lt)) break;
      buf.push(lt);
      i++;
    }
    if (buf.length) out.push("<p>" + inline(buf.join(" ")) + "</p>");
  }
  return out.join("\n");
}

/* ---------- app ---------- */
function buildNav() {
  const nav = document.getElementById("nav");
  nav.innerHTML = "";
  for (const d of DOCS) {
    const a = document.createElement("a");
    a.className = "doc";
    a.href = "#" + d.path;
    a.innerHTML = d.label + "<small>" + d.blurb + "</small>";
    if (d.path === current) a.setAttribute("aria-current", "true");
    nav.appendChild(a);
    if (d.path === current) {
      const box = document.createElement("div");
      box.className = "sections";
      box.id = "sections";
      nav.appendChild(box);
    }
  }
}

function buildSections() {
  const box = document.getElementById("sections");
  if (!box) return;
  box.innerHTML = "";
  document.querySelectorAll("#doc h2").forEach(h => {
    const a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent;
    a.addEventListener("click", e => {
      e.preventDefault();
      h.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    box.appendChild(a);
  });
}

const byBase = {};
DOCS.forEach(d => { byBase[d.path.split("/").pop()] = d.path; });

async function load(path, keepScroll) {
  const y = keepScroll ? window.scrollY : 0;
  const res = await fetch("/raw/" + path + "?t=" + Date.now());
  document.getElementById("doc").innerHTML = render(await res.text());
  buildNav();
  buildSections();
  document.querySelectorAll("#doc a").forEach(a => {
    const href = a.getAttribute("href") || "";
    if (href.endsWith(".md")) {
      const target = byBase[href.split("/").pop()];
      if (target) {
        a.addEventListener("click", e => { e.preventDefault(); location.hash = target; });
      } else {
        a.removeAttribute("href");
        a.style.color = "var(--ink-soft)";
        a.title = "Path in the repo, not part of this preview";
      }
    }
  });
  window.scrollTo(0, y);
}

window.addEventListener("hashchange", () => {
  current = location.hash.slice(1) || DOCS[0].path;
  load(current, false);
});

let toastTimer;
function toast() {
  const el = document.getElementById("toast");
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 1100);
}

async function poll() {
  try {
    const s = await (await fetch("/mtimes?t=" + Date.now())).json();
    const key = JSON.stringify(s);
    if (stamps !== null && stamps !== key) { await load(current, true); toast(); }
    stamps = key;
  } catch (e) { /* server stopped */ }
}

buildNav();
load(current, false);
setInterval(poll, 700);
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            docs = [{"path": p, "label": l, "blurb": b} for p, l, b in DOCS]
            return self._send(PAGE.replace("__DOCS__", json.dumps(docs)), "text/html")

        if path == "/mtimes":
            stamps = {}
            for p, _, _ in DOCS:
                f = ROOT / p
                stamps[p] = f.stat().st_mtime if f.exists() else 0
            return self._send(json.dumps(stamps), "application/json")

        if path.startswith("/raw/"):
            rel = path[len("/raw/"):]
            target = (ROOT / rel).resolve()
            if ROOT in target.parents and target.suffix == ".md" and target.exists():
                return self._send(target.read_text(encoding="utf-8"), "text/plain")
            return self._send("# Not found\n\n`%s`" % rel, "text/plain")

        self.send_error(404)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    url = "http://127.0.0.1:%d/" % PORT
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        print("Staged docs preview → %s" % url)
        print("Edit the .md files; the page reloads on save. Ctrl-C to stop.")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
