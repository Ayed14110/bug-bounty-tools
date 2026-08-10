#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ayed Smart API Hunter - Red Team Flow Edition (Mobile / Termux)
===============================================================

Authorized bug bounty reconnaissance + low-risk HTTP/API baseline automation,
organized as a staged red-team engagement flow.

Engagement flow (phases):
  0. Engagement setup  - authorization ack, scope, output location, formats
  1. Recon (passive)   - katana / gau / waybackurls / robots / sitemap
  2. Enumeration        - native HTML crawl + JS endpoint extraction
  3. Attack surface     - scoring, classification, risk tiering
  4. Active baseline    - low-risk curl checks (GET/HEAD/OPTIONS/CORS/...)
  5. Analysis           - flags, prioritization, manual-test playbook
  6. Reporting/handoff  - md / json / csv / html / playbook / curl evidence

Discovery:
  - Katana / gau / waybackurls (if installed)
  - robots.txt, sitemap.xml
  - HTML links / forms / scripts
  - JavaScript API endpoint strings

Checks (all low-risk, non-destructive):
  - GET baseline, HEAD, OPTIONS
  - CORS GET + preflight
  - security headers, cookie flags, redirects
  - JSON/API detection
  - harmless parameter reflection markers
  - harmless redirect-parameter probes

Safety:
  - strict same-host by default
  - low request rate
  - no brute force / no auth bypass
  - no destructive DELETE/PUT/PATCH
  - no state-changing POST fuzzing
  - no SQLi/XSS/SSRF/RCE exploitation

Use only on systems you are explicitly authorized to test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
import os
import sys
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode, unquote
import xml.etree.ElementTree as ET

VERSION = "3.0-redteam"
UA = f"Ayed-Smart-API-Hunter/{VERSION} authorized-security-testing"
DEFAULT_DELAY = 1.0
MIN_DELAY = 0.50
DEFAULT_TIMEOUT = 15
MAX_RESPONSE_BODY = 1_500_000
MAX_JS_BODY = 1_000_000
MAX_PARAMS_PER_URL = 4

VALID_FORMATS = ("md", "json", "csv", "html", "playbook", "evidence", "urls", "api")
DEFAULT_FORMATS = ("md", "json", "csv", "html", "playbook", "evidence", "urls", "api")

API_HINTS = (
    "/api/", "/api?", "/v1/", "/v2/", "/v3/", "/graphql", "/gql",
    "/rest/", "/ajax", "/rpc", "/json", "/openapi", "/swagger",
)

HIGH_INTEREST_WORDS = (
    "admin", "account", "accounts", "user", "users", "profile", "profiles",
    "order", "orders", "invoice", "invoices", "payment", "payments",
    "wallet", "balance", "transaction", "transactions",
    "upload", "download", "export", "import", "file", "files",
    "callback", "webhook", "redirect", "returnurl", "return_url",
    "oauth", "sso", "token", "auth", "login", "logout", "register",
    "reset", "password", "verify", "verification",
    "internal", "private", "debug", "graphql", "swagger", "openapi",
)

PARAM_INTEREST = (
    "id", "uid", "user", "userid", "user_id", "account", "accountid",
    "account_id", "order", "orderid", "order_id", "invoice", "invoiceid",
    "file", "filename", "path", "url", "uri", "redirect", "redirect_uri",
    "return", "returnurl", "return_url", "next", "callback", "webhook",
    "page", "q", "query", "search", "email", "role", "admin", "token",
)

REDIRECT_PARAMS = {
    "url", "uri", "redirect", "redirect_url", "redirect_uri",
    "return", "returnurl", "return_url", "next", "continue", "dest",
    "destination", "callback", "callback_url",
}

STATIC_EXTS = {
    ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".avi", ".mov",
    ".zip", ".rar", ".7z", ".tar", ".gz",
}
JS_EXTS = {".js", ".mjs", ".cjs"}

SAFE_COMMON_PATHS = (
    "/robots.txt",
    "/sitemap.xml",
    "/.well-known/security.txt",
    "/openapi.json",
    "/swagger.json",
    "/api-docs",
    "/swagger-ui/",
)

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def c(text: str, color: str) -> str:
    codes = {
        "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
        "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
        "bold": "\033[1m", "dim": "\033[2m", "reset": "\033[0m",
    }
    return f"{codes.get(color, '')}{text}{codes['reset']}"

def info(msg: str): print(c(f"[+] {msg}", "cyan"), flush=True)
def ok(msg: str): print(c(f"[✓] {msg}", "green"), flush=True)
def warn(msg: str): print(c(f"[!] {msg}", "yellow"), flush=True)
def fail(msg: str): print(c(f"[x] {msg}", "red"), flush=True)
def die(msg: str, code: int = 1): fail(msg); raise SystemExit(code)

def phase_banner(number, title: str, subtitle: str = ""):
    line = "═" * 62
    print()
    print(c(line, "blue"))
    label = f" PHASE {number} · {title} " if number != "" else f" {title} "
    print(c(label, "bold"))
    if subtitle:
        print(c(f" {subtitle}", "dim"))
    print(c(line, "blue"), flush=True)

# ---------------------------------------------------------------------------
# Interactive prompt helpers
# ---------------------------------------------------------------------------

def interactive_available() -> bool:
    """True only when we can actually read answers from a human."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False

def ask(prompt: str, default: str = "", *, allow_empty: bool = True) -> str:
    """Ask a free-text question. Falls back to default when non-interactive."""
    if not interactive_available():
        return default
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(c(f"    ? {prompt}{suffix}: ", "yellow")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not raw:
            if default or allow_empty:
                return default
            warn("A value is required.")
            continue
        return raw

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    if not interactive_available():
        return default
    d = "Y/n" if default else "y/N"
    while True:
        try:
            raw = input(c(f"    ? {prompt} [{d}]: ", "yellow")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        warn("Please answer y or n.")

def expand_path(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw.strip()))).resolve()

# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------

def normalize_target(raw: str) -> str:
    raw = raw.strip()
    if not re.match(r"^https?://", raw, re.I):
        raw = "https://" + raw
    p = urlsplit(raw)
    if p.scheme not in ("http", "https") or not p.hostname:
        die("Invalid target. Example: https://example.com")
    return urlunsplit((p.scheme.lower(), p.netloc, p.path or "/", p.query, ""))

def host_key(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")

def port_key(url: str):
    p = urlsplit(url)
    if p.port:
        return p.port
    return 443 if p.scheme == "https" else 80

def same_host(target: str, candidate: str) -> bool:
    return host_key(target) == host_key(candidate)

def same_origin(target: str, candidate: str) -> bool:
    a, b = urlsplit(target), urlsplit(candidate)
    return a.scheme.lower() == b.scheme.lower() and host_key(target) == host_key(candidate) and port_key(target) == port_key(candidate)

def canonicalize(url: str, base: str | None = None) -> str | None:
    try:
        if base:
            url = urljoin(base, url)
        url = html.unescape(url.strip())
        if not url or url.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
            return None
        p = urlsplit(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return None
        path = re.sub(r"/{2,}", "/", p.path or "/")
        return urlunsplit((p.scheme.lower(), p.netloc, path, p.query, ""))
    except Exception:
        return None

def ext_of(url: str) -> str:
    name = urlsplit(url).path.lower().rsplit("/", 1)[-1]
    return "" if "." not in name else "." + name.rsplit(".", 1)[-1]

def is_static(url: str) -> bool: return ext_of(url) in STATIC_EXTS
def is_js(url: str) -> bool: return ext_of(url) in JS_EXTS

def base_origin(target: str) -> str:
    p = urlsplit(target)
    return urlunsplit((p.scheme, p.netloc, "", "", ""))

# ---------------------------------------------------------------------------
# Subprocess / external tools
# ---------------------------------------------------------------------------

def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None

def run_process(cmd: list[str], *, stdin_text: str | None = None, timeout: int = 180):
    try:
        p = subprocess.run(cmd, input=stdin_text, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", e.stderr or "timeout"
    except Exception as e:
        return 1, "", str(e)

def collect_urls_from_output(text: str, base: str | None = None) -> list[str]:
    found = []
    for raw in re.findall(r"https?://[^\s<>'\"\]\)]+", text or "", re.I):
        raw = raw.rstrip(".,;")
        u = canonicalize(raw, base)
        if u:
            found.append(u)
    return found

def run_katana(target: str, depth: int):
    meta = {"tool": "katana", "available": tool_exists("katana"), "command": None, "stderr": ""}
    if not meta["available"]:
        return [], meta
    commands = [
        ["katana", "-u", target, "-d", str(depth), "-jc", "-jsl", "-silent"],
        ["katana", "-u", target, "-d", str(depth), "-jc", "-silent"],
        ["katana", "-u", target, "-d", str(depth), "-silent"],
    ]
    rc = 1
    for cmd in commands:
        meta["command"] = " ".join(shlex.quote(x) for x in cmd)
        rc, out, err = run_process(cmd, timeout=300)
        meta["stderr"] = str(err)[-3000:]
        urls = collect_urls_from_output(str(out))
        if urls:
            meta["returncode"] = rc
            return urls, meta
    meta["returncode"] = rc
    return [], meta

def run_gau(target: str):
    meta = {"tool": "gau", "available": tool_exists("gau"), "command": None, "stderr": ""}
    if not meta["available"]:
        return [], meta
    host = host_key(target)
    rc = 1
    for cmd in (["gau", "--threads", "2", host], ["gau", host]):
        meta["command"] = " ".join(shlex.quote(x) for x in cmd)
        rc, out, err = run_process(cmd, timeout=240)
        meta["stderr"] = str(err)[-3000:]
        urls = collect_urls_from_output(str(out))
        if urls:
            meta["returncode"] = rc
            return urls, meta
    meta["returncode"] = rc
    return [], meta

def run_waybackurls(target: str):
    meta = {"tool": "waybackurls", "available": tool_exists("waybackurls"), "command": None, "stderr": ""}
    if not meta["available"]:
        return [], meta
    host = host_key(target)
    meta["command"] = f"printf '%s\\n' {shlex.quote(host)} | waybackurls"
    rc, out, err = run_process(["waybackurls"], stdin_text=host + "\n", timeout=240)
    meta["stderr"] = str(err)[-3000:]
    meta["returncode"] = rc
    return collect_urls_from_output(str(out)), meta

# ---------------------------------------------------------------------------
# HTTP via curl
# ---------------------------------------------------------------------------

def _parse_header_blocks(text: str):
    blocks, current = [], []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("HTTP/"):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            if line == "":
                blocks.append(current)
                current = []
            else:
                current.append(line)
    if current:
        blocks.append(current)
    return blocks

def parse_final_headers(text: str):
    blocks = _parse_header_blocks(text)
    if not blocks:
        return "", {}
    block = blocks[-1]
    status = block[0].strip()
    headers = {}
    for line in block[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers.setdefault(k.strip().lower(), []).append(v.strip())
    return status, headers

def first_header(resp: dict, name: str):
    vals = resp.get("headers", {}).get(name.lower(), [])
    return vals[-1] if vals else None

def sanitize_header_for_log(h: str) -> str:
    if re.match(r"(?i)^\s*(authorization|cookie|proxy-authorization)\s*:", h):
        return h.split(":", 1)[0] + ": <redacted>"
    return h

def curl_request(url: str, method: str = "GET", *, headers=None, timeout=DEFAULT_TIMEOUT, follow=False, max_body=MAX_RESPONSE_BODY):
    if not tool_exists("curl"):
        die("curl required. Termux: pkg install curl")
    headers = headers or []
    with tempfile.TemporaryDirectory(prefix="ayed_bbp_") as td:
        hfile, bfile = Path(td) / "headers.txt", Path(td) / "body.bin"
        writeout = (
            '{"http_code":%{http_code},"time_total":%{time_total},'
            '"time_connect":%{time_connect},"time_starttransfer":%{time_starttransfer},'
            '"size_download":%{size_download},"content_type":"%{content_type}",'
            '"url_effective":"%{url_effective}","remote_ip":"%{remote_ip}",'
            '"http_version":"%{http_version}","num_redirects":%{num_redirects},'
            '"ssl_verify_result":%{ssl_verify_result}}'
        )
        cmd = ["curl", "-sS", "--compressed", "--max-time", str(timeout), "--connect-timeout", "8",
               "-A", UA, "-X", method, "-D", str(hfile), "-o", str(bfile), "-w", writeout]
        if follow:
            cmd += ["-L", "--max-redirs", "5"]
        for h in headers:
            cmd += ["-H", h]
        cmd.append(url)
        started = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = round(time.time() - started, 3)
        htext = hfile.read_text(errors="replace") if hfile.exists() else ""
        body_bytes = bfile.read_bytes()[:max_body] if bfile.exists() else b""
        status_line, parsed_headers = parse_final_headers(htext)
        try:
            meta = json.loads(p.stdout.strip()) if p.stdout.strip() else {}
        except Exception:
            meta = {"raw_writeout": p.stdout.strip()}
        logged_cmd = list(cmd)
        for i, x in enumerate(logged_cmd):
            if i > 0 and logged_cmd[i - 1] == "-H":
                logged_cmd[i] = sanitize_header_for_log(x)
        return {
            "url": url, "method": method, "status_line": status_line,
            "headers": parsed_headers,
            "body_text": body_bytes.decode("utf-8", errors="replace"),
            "body_sha256": hashlib.sha256(body_bytes).hexdigest(),
            "body_sample_bytes": len(body_bytes), "meta": meta,
            "returncode": p.returncode, "stderr": p.stderr.strip(),
            "wall_time": wall,
            "command": " ".join(shlex.quote(str(x)) for x in logged_cmd),
        }

# ---------------------------------------------------------------------------
# HTML / sitemap / robots parsing
# ---------------------------------------------------------------------------

class LinkFormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls, self.forms, self.scripts = [], [], []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag in ("a", "link") and d.get("href"):
            self.urls.append(d["href"])
        if tag in ("img", "script", "iframe", "source") and d.get("src"):
            self.urls.append(d["src"])
        if tag == "script" and d.get("src"):
            self.scripts.append(d["src"])
        if tag == "form":
            self.forms.append({"action": d.get("action", ""), "method": (d.get("method") or "GET").upper()})

def parse_html(base_url: str, body: str):
    parser = LinkFormParser()
    try:
        parser.feed(body)
    except Exception:
        pass
    urls, scripts, forms = [], [], []
    for raw in parser.urls:
        u = canonicalize(raw, base_url)
        if u: urls.append(u)
    for raw in parser.scripts:
        u = canonicalize(raw, base_url)
        if u: scripts.append(u)
    for f in parser.forms:
        u = canonicalize(f.get("action") or base_url, base_url)
        if u: forms.append({"url": u, "method": f["method"]})
    return urls, scripts, forms

def parse_sitemap(body: str, base_url: str):
    urls = []
    try:
        root = ET.fromstring(body)
        for elem in root.iter():
            if elem.tag.lower().endswith("loc") and elem.text:
                u = canonicalize(elem.text.strip(), base_url)
                if u: urls.append(u)
    except Exception:
        for value in re.findall(r"<loc>\s*([^<]+)\s*</loc>", body, re.I):
            u = canonicalize(value.strip(), base_url)
            if u: urls.append(u)
    return urls

def parse_robots(body: str, base_url: str):
    urls = []
    for line in body.splitlines():
        m = re.match(r"(?i)^(allow|disallow|sitemap)\s*:\s*(.+)$", line.strip())
        if not m: continue
        value = m.group(2).strip()
        if not value or value == "/": continue
        u = canonicalize(value, base_url)
        if u: urls.append(u)
    return urls

ABS_URL_RE = re.compile(r"""(?i)\bhttps?://[a-z0-9._~:/?#\[\]@!$&()*+,;=%-]+""")
REL_API_RE = re.compile(
    r"""(?ix)["'`](/
        (?:
            api(?:/|$)|v[0-9]+(?:/|$)|graphql(?:/|$)|gql(?:/|$)|rest(?:/|$)|
            ajax(?:/|$)|rpc(?:/|$)|auth(?:/|$)|oauth(?:/|$)|users?(?:/|$)|
            accounts?(?:/|$)|orders?(?:/|$)|payments?(?:/|$)|files?(?:/|$)|
            uploads?(?:/|$)|downloads?(?:/|$)|admin(?:/|$)
        )[^"'`<>\s]{0,500}
    )["'`]"""
)

def extract_js_endpoints(js_url: str, body: str):
    out = []
    for raw in ABS_URL_RE.findall(body):
        u = canonicalize(raw)
        if u: out.append(u)
    for raw in REL_API_RE.findall(body):
        u = canonicalize(raw, js_url)
        if u: out.append(u)
    return out

# ---------------------------------------------------------------------------
# Scoring / classification / attack-surface tiering
# ---------------------------------------------------------------------------

def query_keys(url: str):
    return [k for k, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True)]

def endpoint_score(url: str):
    low = unquote(url).lower()
    p = urlsplit(url)
    score, reasons = 0, []
    if any(h in low for h in API_HINTS):
        score += 25; reasons.append("api-like")
    for word in HIGH_INTEREST_WORDS:
        if re.search(rf"(^|[/_.?=&-]){re.escape(word)}([/_.?=&-]|$)", low):
            score += 6; reasons.append(word)
    keys = [k.lower() for k in query_keys(url)]
    if keys:
        score += min(20, len(keys) * 3); reasons.append(f"{len(keys)}-params")
    for k in keys:
        if k in PARAM_INTEREST:
            score += 5; reasons.append(f"param:{k}")
    if p.path.lower().endswith((".json", ".xml")):
        score += 7; reasons.append("structured-file")
    if is_js(url):
        score -= 10; reasons.append("javascript")
    elif is_static(url):
        score -= 30; reasons.append("static")
    score += min(len([x for x in p.path.split("/") if x]), 5)
    return max(score, 0), list(dict.fromkeys(reasons))

def classify(url: str):
    low = unquote(url).lower()
    tags = []
    if any(h in low for h in API_HINTS): tags.append("api")
    if "graphql" in low or "/gql" in low: tags.append("graphql")
    if "swagger" in low or "openapi" in low or "api-docs" in low: tags.append("api-docs")
    if any(x in low for x in ("auth", "login", "logout", "oauth", "sso", "token")): tags.append("auth")
    if any(x in low for x in ("account", "profile", "/user", "/users")): tags.append("identity")
    if any(x in low for x in ("admin", "internal", "private")): tags.append("privileged")
    if any(x in low for x in ("upload", "import")): tags.append("ingest")
    if any(x in low for x in ("download", "export", "file")): tags.append("file")
    if any(x in low for x in ("callback", "webhook")): tags.append("callback")
    if any(x in low for x in ("redirect", "returnurl", "return_url", "next=")): tags.append("redirect")
    if query_keys(url): tags.append("parameters")
    if is_js(url): tags.append("javascript")
    return list(dict.fromkeys(tags))

def risk_tier(score: int, tags: list) -> str:
    """Red-team style prioritization tier for the attack surface."""
    tset = set(tags)
    if tset & {"privileged", "graphql"} and (tset & {"api", "parameters", "auth", "identity"}):
        return "CRITICAL"
    if score >= 30 or (tset & {"api", "auth", "identity", "api-docs"} and tset & {"parameters"}):
        return "HIGH"
    if score >= 18 or (tset & {"api", "auth", "identity", "file", "callback", "redirect", "ingest", "api-docs"}):
        return "MEDIUM"
    if score >= 8 or "parameters" in tset:
        return "LOW"
    return "INFO"

TIER_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
TIER_COLOR = {"CRITICAL": "red", "HIGH": "magenta", "MEDIUM": "yellow", "LOW": "cyan", "INFO": "dim"}

# ---------------------------------------------------------------------------
# Response analysis
# ---------------------------------------------------------------------------

def analyze_security_headers(resp: dict):
    h = resp["headers"]
    expected = {
        "strict-transport-security": "HSTS",
        "content-security-policy": "CSP",
        "x-content-type-options": "X-Content-Type-Options",
        "referrer-policy": "Referrer-Policy",
        "permissions-policy": "Permissions-Policy",
        "cross-origin-opener-policy": "COOP",
    }
    present, missing = [], []
    for k, label in expected.items():
        (present if k in h else missing).append(label)
    return {"present": present, "missing": missing}

def analyze_cookies(resp: dict):
    out = []
    for raw in resp.get("headers", {}).get("set-cookie", []):
        low = raw.lower()
        name = raw.split("=", 1)[0].strip() if "=" in raw else "<unknown>"
        out.append({"name": name, "secure": "; secure" in low, "httponly": "; httponly" in low, "samesite": "samesite=" in low})
    return out

def analyze_cors(resp: dict, origin: str):
    acao, acac = first_header(resp, "access-control-allow-origin"), first_header(resp, "access-control-allow-credentials")
    out = {"probe_origin": origin, "acao": acao, "acac": acac, "manual_review": False, "reason": ""}
    if acao == origin and (acac or "").lower() == "true":
        out["manual_review"] = True
        out["reason"] = "Probe Origin reflected with credentials; verify whether authenticated sensitive data is readable."
    elif acao == origin:
        out["manual_review"] = True
        out["reason"] = "Probe Origin reflected; context requires manual review."
    elif acao == "*" and (acac or "").lower() == "true":
        out["manual_review"] = True
        out["reason"] = "Wildcard ACAO plus credentials header observed; verify browser behavior/context manually."
    return out

def looks_json(resp: dict):
    ct = (first_header(resp, "content-type") or "").lower()
    body = resp.get("body_text", "").lstrip()
    return "json" in ct or body.startswith("{") or body.startswith("[")

def auth_signal(resp: dict):
    status = int(resp.get("meta", {}).get("http_code") or 0)
    body = resp.get("body_text", "").lower()[:12000]
    if status == 401 or first_header(resp, "www-authenticate"): return "authentication-required"
    if status == 403: return "forbidden"
    if any(x in body for x in ("unauthorized", "authentication required", "please login", "please log in")): return "auth-message"
    return "none"

def method_advertisement(options_resp: dict):
    values = []
    for name in ("allow", "access-control-allow-methods"):
        for v in options_resp.get("headers", {}).get(name, []):
            values += [x.strip().upper() for x in v.split(",") if x.strip()]
    return sorted(set(values))

def safe_parameter_probes(url: str, delay: float):
    params = parse_qsl(urlsplit(url).query, keep_blank_values=True)
    probes = []
    for idx, (key, original) in enumerate(params[:MAX_PARAMS_PER_URL]):
        marker = "ayed_probe_" + uuid.uuid4().hex[:10]
        changed = list(params)
        changed[idx] = (key, marker)
        p = urlsplit(url)
        test_url = urlunsplit((p.scheme, p.netloc, p.path, urlencode(changed, doseq=True), ""))
        time.sleep(delay)
        resp = curl_request(test_url, "GET")
        loc = first_header(resp, "location") or ""
        probes.append({
            "parameter": key, "original": original[:160], "marker": marker,
            "status": resp.get("meta", {}).get("http_code"),
            "body_reflection": marker in resp.get("body_text", ""),
            "location_reflection": marker in loc,
            "content_type": first_header(resp, "content-type"),
            "body_sha256": resp.get("body_sha256"),
            "command": resp.get("command"),
        })
    return probes

def safe_redirect_probe(url: str, delay: float):
    params = parse_qsl(urlsplit(url).query, keep_blank_values=True)
    results = []
    for idx, (key, original) in enumerate(params[:MAX_PARAMS_PER_URL]):
        if key.lower() not in REDIRECT_PARAMS: continue
        changed = list(params)
        test_value = "https://example.invalid/ayed-bbp-probe"
        changed[idx] = (key, test_value)
        p = urlsplit(url)
        test_url = urlunsplit((p.scheme, p.netloc, p.path, urlencode(changed), ""))
        time.sleep(delay)
        resp = curl_request(test_url, "GET", follow=False)
        location = first_header(resp, "location") or ""
        results.append({
            "parameter": key, "original": original[:160], "probe_value": test_value,
            "status": resp.get("meta", {}).get("http_code"), "location": location[:1000],
            "external_location_observed": location.startswith(test_value),
            "command": resp.get("command"),
        })
    return results

def scan_endpoint(url: str, delay: float):
    score, reasons = endpoint_score(url)
    tags = classify(url)
    result = {
        "url": url, "score": score, "score_reasons": reasons,
        "tags": tags, "tier": risk_tier(score, tags), "tests": {},
    }
    info(f"GET {url}")
    base = curl_request(url, "GET")
    result["tests"]["get"] = base
    result["json_like"] = looks_json(base)
    result["auth_signal"] = auth_signal(base)
    result["security_headers"] = analyze_security_headers(base)
    result["cookies"] = analyze_cookies(base)

    time.sleep(delay); head = curl_request(url, "HEAD"); result["tests"]["head"] = head
    time.sleep(delay); opt = curl_request(url, "OPTIONS"); result["tests"]["options"] = opt
    result["advertised_methods"] = method_advertisement(opt)

    time.sleep(delay)
    origin = "https://ayed-bbp-probe.invalid"
    cors = curl_request(url, "GET", headers=[f"Origin: {origin}"])
    result["tests"]["cors_get"] = cors
    result["cors"] = analyze_cors(cors, origin)

    time.sleep(delay)
    preflight = curl_request(url, "OPTIONS", headers=[
        f"Origin: {origin}",
        "Access-Control-Request-Method: GET",
        "Access-Control-Request-Headers: X-Ayed-BBP-Probe",
    ])
    result["tests"]["cors_preflight"] = preflight

    result["parameter_probes"] = safe_parameter_probes(url, delay) if query_keys(url) else []
    result["redirect_probes"] = safe_redirect_probe(url, delay) if query_keys(url) else []
    result["summary"] = {
        "http": base.get("meta", {}).get("http_code"),
        "size": base.get("meta", {}).get("size_download"),
        "time": base.get("meta", {}).get("time_total"),
        "content_type": first_header(base, "content-type"),
        "location": first_header(base, "location"),
        "server": first_header(base, "server"),
    }
    result["flags"] = finding_flags(result)
    return result

# ---------------------------------------------------------------------------
# Discovery store
# ---------------------------------------------------------------------------

class DiscoveryStore:
    def __init__(self, target: str, max_urls: int):
        self.target, self.max_urls, self.urls, self.forms = target, max_urls, {}, []
    def add(self, url, source: str):
        if not url: return False
        u = canonicalize(url)
        if not u or not same_host(self.target, u): return False
        if len(self.urls) >= self.max_urls and u not in self.urls: return False
        self.urls.setdefault(u, set()).add(source)
        return True
    def add_many(self, urls, source: str):
        for u in urls: self.add(u, source)
    def sorted(self):
        items = []
        for u, sources in self.urls.items():
            score, reasons = endpoint_score(u)
            tags = classify(u)
            items.append({
                "url": u, "sources": sorted(sources), "score": score,
                "score_reasons": reasons, "tags": tags, "tier": risk_tier(score, tags),
            })
        return sorted(items, key=lambda x: (TIER_ORDER.get(x["tier"], 9), -x["score"], x["url"]))

def discover_native(target: str, store: DiscoveryStore, delay: float, native_pages: int):
    origin = base_origin(target)
    for p in SAFE_COMMON_PATHS: store.add(origin + p, "common")
    for u in [origin + "/robots.txt", origin + "/sitemap.xml"]:
        time.sleep(delay)
        resp = curl_request(u, "GET")
        body = resp.get("body_text", "")
        store.add_many(parse_robots(body, origin + "/") if u.endswith("robots.txt") else parse_sitemap(body, origin + "/"),
                       "robots" if u.endswith("robots.txt") else "sitemap")
    store.add(target, "seed")
    queue, seen = [target], set()
    while queue and len(seen) < native_pages:
        u = queue.pop(0)
        if u in seen or is_static(u): continue
        seen.add(u)
        time.sleep(delay)
        resp = curl_request(u, "GET")
        ct = (first_header(resp, "content-type") or "").lower()
        if "html" not in ct and "<html" not in resp.get("body_text", "").lower()[:3000]:
            continue
        links, scripts, forms = parse_html(u, resp["body_text"])
        before = set(store.urls.keys())
        store.add_many(links, "html"); store.add_many(scripts, "html-script")
        for f in forms:
            if same_host(target, f["url"]):
                store.forms.append(f); store.add(f["url"], f"form:{f['method']}")
        queue.extend([x for x in store.urls.keys() if x not in before and not is_static(x)][:20])

def discover_js(target: str, store: DiscoveryStore, delay: float, max_js: int):
    js_urls = sorted([u for u in store.urls if is_js(u)], key=lambda u: (not same_origin(target, u), u))[:max_js]
    for js in js_urls:
        time.sleep(delay)
        resp = curl_request(js, "GET", max_body=MAX_JS_BODY)
        if int(resp.get("meta", {}).get("http_code") or 0) >= 400: continue
        store.add_many(extract_js_endpoints(js, resp.get("body_text", "")), "javascript")

# ---------------------------------------------------------------------------
# Findings + playbook
# ---------------------------------------------------------------------------

def finding_flags(scan: dict):
    flags = []
    if scan.get("cors", {}).get("manual_review"): flags.append("CORS_REVIEW")
    if any(x.get("body_reflection") for x in scan.get("parameter_probes", [])): flags.append("REFLECTION")
    if any(x.get("location_reflection") for x in scan.get("parameter_probes", [])): flags.append("LOCATION_REFLECTION")
    if any(x.get("external_location_observed") for x in scan.get("redirect_probes", [])): flags.append("REDIRECT_REVIEW")
    if scan.get("json_like"): flags.append("JSON_API")
    if scan.get("auth_signal") != "none": flags.append(scan["auth_signal"].upper().replace("-", "_"))
    if set(scan.get("advertised_methods", [])) & {"PUT", "PATCH", "DELETE"}: flags.append("STATE_METHODS_ADVERTISED")
    missing = scan.get("security_headers", {}).get("missing", [])
    if missing:
        flags.append("MISSING_SECURITY_HEADERS")
    for ck in scan.get("cookies", []):
        if not ck.get("secure") or not ck.get("httponly"):
            flags.append("WEAK_COOKIE_FLAGS"); break
    return flags

# Red-team next-step suggestions keyed by flag / tag. Verification only, no exploitation.
# Technique numbers (T#) map to RED_TEAM_METHODOLOGY.md.
PLAYBOOK_RULES = {
    "JSON_API": "[T1/T3/T22] Map request/response schema and object identifiers (id/uid/order_id). Manually test IDOR/BOLA and mass-assignment using two accounts you control.",
    "CORS_REVIEW": "[T28] Reproduce the cross-origin read in a browser with an attacker-controlled origin and a test account; confirm authenticated sensitive data is actually readable before reporting.",
    "REFLECTION": "[T11/T13] Inspect the reflection context (HTML/JS/attribute) manually for missing output encoding / template evaluation ({{7*7}}); do not run automated payloads against production.",
    "LOCATION_REFLECTION": "[T21] Check whether the reflected parameter drives the Location header and whether external hosts are honored (open-redirect path).",
    "REDIRECT_REVIEW": "[T21/T7] Confirm the external redirect, then assess chaining into OAuth code theft or SSRF.",
    "STATE_METHODS_ADVERTISED": "[T2/T25] PUT/PATCH/DELETE advertised only — do NOT invoke blindly. Test BFLA and method-override (X-HTTP-Method-Override) semantics on resources you own.",
    "AUTHENTICATION_REQUIRED": "[T23/T24] Compare unauth vs auth exposure; look for shadow/older API versions (v1/internal) and rate-limit gaps on the auth flow.",
    "FORBIDDEN": "[T5] Try path/method/header variations (X-Original-URL, trailing dot, case, .json) that change the 403 — documentation only, with your own accounts.",
    "MISSING_SECURITY_HEADERS": "[T18/T19] Missing CSP/HSTS/etc. — assess cache poisoning and Host-header injection impact in context, not as a bare finding.",
    "WEAK_COOKIE_FLAGS": "[T9/T10] Session/auth cookies missing Secure/HttpOnly/SameSite; tie to session theft/CSRF or 2FA-step separation before reporting.",
}
TAG_PLAYBOOK = {
    "graphql": "[T22] Attempt introspection (if scope allows), then test aliasing/batching for rate-limit bypass and BOLA on node ids you own.",
    "api-docs": "[T23] Read the exposed OpenAPI/Swagger spec to enumerate endpoints/params/auth for targeted manual testing and version drift.",
    "privileged": "[T2] High priority: verify whether admin/internal functionality is reachable by a low-privilege account (BFLA).",
    "auth": "[T6/T7/T8] Review the auth flow for logic gaps: JWT handling (alg/none/kid), OAuth redirect_uri/state, password-reset host-header — manually, with your own accounts.",
    "identity": "[T1] Prime target for IDOR/BOLA: enumerate identifiers and confirm object-level authorization with two controlled accounts.",
    "file": "[T12/T27] Review file/export endpoints for path traversal, SSRF via url params, and exposed source/secret leaks — scoped and non-destructive.",
    "ingest": "[T3] Review upload/import handling (type/size/content validation, mass assignment) using benign test files only.",
    "callback": "[T12/T7] Callback/webhook params are SSRF and OAuth-chain candidates: point at a collaborator host you own and observe.",
    "redirect": "[T21] Open-redirect candidate: confirm external Location, then assess OAuth/SSRF chaining.",
}

def build_playbook(scans: list):
    entries = []
    for s in sorted(scans, key=lambda x: (TIER_ORDER.get(x.get("tier", "INFO"), 9), -x.get("score", 0))):
        steps = []
        for fl in s.get("flags", []):
            if fl in PLAYBOOK_RULES:
                steps.append(PLAYBOOK_RULES[fl])
        for tag in s.get("tags", []):
            if tag in TAG_PLAYBOOK:
                steps.append(TAG_PLAYBOOK[tag])
        steps = list(dict.fromkeys(steps))
        if not steps and s.get("tier") in ("CRITICAL", "HIGH", "MEDIUM"):
            steps.append("Manually enumerate parameters/responses and check authorization with accounts you control.")
        if steps:
            entries.append({
                "url": s["url"], "tier": s.get("tier", "INFO"),
                "score": s.get("score", 0), "flags": s.get("flags", []),
                "tags": s.get("tags", []), "steps": steps,
            })
    return entries

# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def compact_response(resp: dict):
    return {
        "url": resp.get("url"), "method": resp.get("method"), "status_line": resp.get("status_line"),
        "headers": resp.get("headers"), "body_sha256": resp.get("body_sha256"),
        "body_sample_bytes": resp.get("body_sample_bytes"), "body_preview": resp.get("body_text", "")[:3000],
        "meta": resp.get("meta"), "returncode": resp.get("returncode"),
        "stderr": resp.get("stderr"), "wall_time": resp.get("wall_time"), "command": resp.get("command"),
    }

def compact_scan(scan: dict):
    x = dict(scan)
    x["tests"] = {k: compact_response(v) for k, v in scan.get("tests", {}).items()}
    return x

def tier_counts(discovered: list):
    counts = {t: 0 for t in TIER_ORDER}
    for d in discovered:
        counts[d.get("tier", "INFO")] = counts.get(d.get("tier", "INFO"), 0) + 1
    return counts

def write_reports(out: Path, data: dict, formats: tuple):
    paths = {}
    discovered = data["discovered"]
    scans = data["scans"]
    scans_by_url = {x["url"]: x for x in scans}
    playbook = data.get("playbook", [])

    if "json" in formats:
        json_path = out / "report.json"
        json_data = dict(data)
        json_data["scans"] = [compact_scan(x) for x in scans]
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json"] = json_path

    if "urls" in formats:
        p = out / "discovered_urls.txt"
        p.write_text("\n".join(x["url"] for x in discovered) + "\n", encoding="utf-8")
        paths["urls"] = p

    if "api" in formats:
        api_candidates = [x for x in discovered if x["tier"] in ("CRITICAL", "HIGH") or {"api", "graphql", "api-docs"} & set(x["tags"]) or x["score"] >= 20]
        p = out / "api_candidates.txt"
        p.write_text("\n".join(x["url"] for x in api_candidates) + ("\n" if api_candidates else ""), encoding="utf-8")
        paths["api"] = p

    if "csv" in formats:
        csv_path = out / "endpoints.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tier", "score", "url", "sources", "tags", "http", "content_type", "auth_signal", "methods", "flags"])
            for d in discovered:
                s = scans_by_url.get(d["url"], {})
                w.writerow([d["tier"], d["score"], d["url"], ",".join(d["sources"]), ",".join(d["tags"]),
                            s.get("summary", {}).get("http", ""), s.get("summary", {}).get("content_type", ""),
                            s.get("auth_signal", ""), ",".join(s.get("advertised_methods", [])),
                            ",".join(s.get("flags", [])) if s else ""])
        paths["csv"] = csv_path

    if "evidence" in formats:
        evidence = out / "curl_evidence.sh"
        commands = ["#!/data/data/com.termux/files/usr/bin/bash", "# Generated curl evidence. Review before running.", "set -u", ""]
        for scan in scans:
            commands.append(f"# [{scan.get('tier')}] {scan['url']}")
            for tname in ("get", "head", "options", "cors_get", "cors_preflight"):
                r = scan.get("tests", {}).get(tname)
                if r and r.get("command"): commands.append(r["command"])
            for p in scan.get("parameter_probes", []) + scan.get("redirect_probes", []):
                if p.get("command"): commands.append(p["command"])
            commands.append("")
        evidence.write_text("\n".join(commands), encoding="utf-8")
        try: evidence.chmod(0o755)
        except Exception: pass
        paths["evidence"] = evidence

    if "playbook" in formats:
        paths["playbook"] = write_playbook_md(out, data, playbook)

    if "md" in formats:
        paths["markdown"] = write_report_md(out, data)

    if "html" in formats:
        paths["html"] = write_report_html(out, data)

    return paths

def write_playbook_md(out: Path, data: dict, playbook: list):
    md = [
        "# Red Team Manual-Test Playbook", "",
        f"- Target: `{data['target']}`",
        f"- Generated: `{data['generated_at']}`",
        f"- Engagement: `{data.get('engagement', {}).get('name', '-')}`",
        "",
        "> Verification-only guidance. Nothing here authorizes destructive testing, "
        "exploitation, or out-of-scope activity. Always test authorization with "
        "accounts and resources you own.", "",
    ]
    md += ["Technique tags `[T#]` reference `RED_TEAM_METHODOLOGY.md` (30 advanced techniques).", ""]
    if not playbook:
        md += ["_No prioritized leads generated._", ""]
    for i, e in enumerate(playbook, 1):
        md += [
            f"## {i}. [{e['tier']}] {e['url']}", "",
            f"- Score: **{e['score']}** · Tags: `{','.join(e['tags']) or '-'}` · Flags: `{','.join(e['flags']) or 'none'}`",
            "", "**Suggested manual verification:**", "",
        ]
        for step in e["steps"]:
            md.append(f"- {step}")
        md.append("")
    p = out / "playbook.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p

def write_report_md(out: Path, data: dict):
    discovered = data["discovered"]
    counts = tier_counts(discovered)
    md = [
        "# Ayed Smart API Hunter Report", "",
        f"- Version: `{data['version']}`", f"- Generated: `{data['generated_at']}`",
        f"- Target: `{data['target']}`", f"- Strict host scope: `{host_key(data['target'])}`",
        f"- Engagement: `{data.get('engagement', {}).get('name', '-')}`",
        f"- Operator: `{data.get('engagement', {}).get('operator', '-')}`",
        f"- Discovered URLs: **{len(discovered)}**", f"- Scanned endpoints: **{len(data['scans'])}**",
        f"- Delay: **{data['delay']}s**", "",
        "**Attack surface by tier:** " + " · ".join(f"{t}: **{counts.get(t,0)}**" for t in ("CRITICAL","HIGH","MEDIUM","LOW","INFO")),
        "",
        "> Leads below require manual verification. A flag is not automatically a vulnerability.", "",
        "## Tool status", "",
    ]
    for tm in data["tool_meta"]:
        md.append(f"- **{tm['tool']}**: " + ("available" if tm.get("available") else "not installed") + (f" — `{tm.get('command')}`" if tm.get("command") else ""))
    md += ["", "## Attack surface — top prioritized endpoints", ""]
    for d in discovered[:60]:
        md.append(f"- `[{d['tier']}]` **{d['score']:>2}** `{d['url']}` — tags: `{','.join(d['tags']) or '-'}` — sources: `{','.join(d['sources'])}`")
    if data.get("forms"):
        md += ["", "## Forms discovered", ""]
        for f in data["forms"][:100]:
            md.append(f"- `{f['method']}` `{f['url']}`")
    md += ["", "## Active baseline results", ""]
    for s in data["scans"]:
        sm, flags = s.get("summary", {}), s.get("flags", [])
        md += [
            f"### [{s.get('tier')}] {s['url']}", "",
            f"- Discovery score: **{s['score']}** ({', '.join(s['score_reasons']) or 'baseline'})",
            f"- Tags: `{','.join(s['tags']) or '-'}`", f"- HTTP: **{sm.get('http')}**",
            f"- Content-Type: `{sm.get('content_type')}`", f"- Size: `{sm.get('size')}` bytes",
            f"- Time: `{sm.get('time')}` sec", f"- Auth signal: `{s.get('auth_signal')}`",
            f"- Advertised methods: `{','.join(s.get('advertised_methods', [])) or 'none'}`",
            f"- Missing security headers: `{','.join(s.get('security_headers', {}).get('missing', [])) or 'none'}`",
            f"- Flags: **{', '.join(flags) or 'none'}**", "",
        ]
        if s.get("cors", {}).get("manual_review"):
            md += [f"- CORS review: {s['cors'].get('reason')}", ""]
        for p in s.get("parameter_probes", []):
            if p.get("body_reflection") or p.get("location_reflection"):
                md += [f"- Parameter `{p['parameter']}` reflected: body={p['body_reflection']} location={p['location_reflection']}", ""]
        for p in s.get("redirect_probes", []):
            if p.get("external_location_observed"):
                md += [f"- Redirect lead `{p['parameter']}` -> `{p['location']}`", ""]
    md += [
        "## Files", "",
        "- `discovered_urls.txt` — all same-host URLs",
        "- `api_candidates.txt` — API/high-interest candidates",
        "- `endpoints.csv` — sortable inventory (with tier)",
        "- `playbook.md` — prioritized manual-test playbook",
        "- `curl_evidence.sh` — exact curl commands",
        "- `report.json` — structured evidence",
        "- `report.html` — visual report",
        "- `RED_TEAM_METHODOLOGY.md` — 30 advanced attacker-mindset techniques (repo root)", "",
        "## Manual verification priority", "",
        "1. JSON/API endpoints with object/account/order identifiers (IDOR/BOLA).",
        "2. CORS leads only when sensitive authenticated data is readable.",
        "3. Redirect leads only when an external Location is actually observed.",
        "4. Advertised PUT/PATCH/DELETE are clues only; do not invoke destructive methods blindly.",
        "5. Test authorization using accounts/resources you control.", "",
    ]
    p = out / "report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p

def write_report_html(out: Path, data: dict):
    esc = html.escape
    discovered = data["discovered"]
    counts = tier_counts(discovered)
    tier_badge = {
        "CRITICAL": "#b91c1c", "HIGH": "#c026d3", "MEDIUM": "#ca8a04",
        "LOW": "#0891b2", "INFO": "#64748b",
    }
    rows = []
    for d in discovered[:200]:
        rows.append(
            f"<tr><td><span class='badge' style='background:{tier_badge.get(d['tier'],'#64748b')}'>{esc(d['tier'])}</span></td>"
            f"<td class='num'>{d['score']}</td><td class='url'>{esc(d['url'])}</td>"
            f"<td>{esc(','.join(d['tags']) or '-')}</td><td>{esc(','.join(d['sources']))}</td></tr>"
        )
    scan_blocks = []
    for s in data["scans"]:
        sm = s.get("summary", {})
        flags = s.get("flags", [])
        scan_blocks.append(
            f"<div class='card'><h3><span class='badge' style='background:{tier_badge.get(s.get('tier'),'#64748b')}'>{esc(s.get('tier',''))}</span> "
            f"{esc(s['url'])}</h3>"
            f"<p class='meta'>HTTP <b>{esc(str(sm.get('http')))}</b> · {esc(str(sm.get('content_type')))} · "
            f"{esc(str(sm.get('size')))} bytes · auth: {esc(str(s.get('auth_signal')))}</p>"
            f"<p>Methods: <code>{esc(','.join(s.get('advertised_methods', [])) or 'none')}</code></p>"
            f"<p>Flags: {''.join(f'<span class=flag>{esc(x)}</span>' for x in flags) or '<span class=dim>none</span>'}</p>"
            f"</div>"
        )
    playbook_blocks = []
    for i, e in enumerate(data.get("playbook", []), 1):
        steps = "".join(f"<li>{esc(st)}</li>" for st in e["steps"])
        playbook_blocks.append(
            f"<div class='card'><h3>{i}. <span class='badge' style='background:{tier_badge.get(e['tier'],'#64748b')}'>{esc(e['tier'])}</span> "
            f"{esc(e['url'])}</h3><ul>{steps}</ul></div>"
        )
    summary_cards = "".join(
        f"<div class='stat'><div class='statnum' style='color:{tier_badge[t]}'>{counts.get(t,0)}</div><div class='statlbl'>{t}</div></div>"
        for t in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>API Hunter Report — {esc(host_key(data['target']))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#0f172a; color:#e2e8f0; }}
  header {{ padding: 24px; background:#1e293b; border-bottom:3px solid #334155; }}
  header h1 {{ margin:0 0 6px; font-size:20px; }}
  header p {{ margin:2px 0; color:#94a3b8; font-size:13px; }}
  main {{ padding: 20px; max-width: 1100px; margin:0 auto; }}
  h2 {{ margin-top:32px; border-bottom:1px solid #334155; padding-bottom:6px; }}
  .stats {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
  .stat {{ background:#1e293b; border-radius:10px; padding:14px 20px; text-align:center; min-width:90px; }}
  .statnum {{ font-size:26px; font-weight:700; }}
  .statlbl {{ font-size:11px; color:#94a3b8; letter-spacing:1px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid #24324a; vertical-align:top; }}
  th {{ color:#94a3b8; font-weight:600; }}
  td.url {{ word-break:break-all; font-family:ui-monospace,monospace; }}
  td.num {{ text-align:right; color:#38bdf8; }}
  .badge {{ color:#fff; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:700; }}
  .card {{ background:#1e293b; border-radius:10px; padding:14px 16px; margin:12px 0; }}
  .card h3 {{ margin:0 0 8px; font-size:14px; word-break:break-all; }}
  .meta {{ color:#94a3b8; font-size:12px; }}
  .flag {{ display:inline-block; background:#334155; color:#fbbf24; padding:2px 7px; border-radius:8px; font-size:11px; margin:2px; }}
  .dim {{ color:#64748b; }}
  code {{ background:#0f172a; padding:1px 5px; border-radius:5px; }}
  .note {{ background:#422006; border:1px solid #a16207; padding:12px; border-radius:8px; color:#fde68a; font-size:13px; }}
  .tablewrap {{ overflow-x:auto; }}
</style></head>
<body>
<header>
  <h1>Ayed Smart API Hunter — Red Team Flow Report</h1>
  <p>Target: <b>{esc(data['target'])}</b> · Scope host: <b>{esc(host_key(data['target']))}</b></p>
  <p>Engagement: {esc(str(data.get('engagement', {}).get('name', '-')))} · Operator: {esc(str(data.get('engagement', {}).get('operator', '-')))}</p>
  <p>Generated: {esc(data['generated_at'])} · Version {esc(data['version'])}</p>
</header>
<main>
  <p class="note">Authorized testing only. Leads require manual verification — a flag is not automatically a vulnerability. No destructive or exploitation testing is implied.</p>
  <h2>Attack surface</h2>
  <div class="stats">{summary_cards}</div>
  <div class="tablewrap"><table>
    <thead><tr><th>Tier</th><th>Score</th><th>URL</th><th>Tags</th><th>Sources</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
  <h2>Manual-test playbook</h2>
  {''.join(playbook_blocks) or '<p class=dim>No prioritized leads.</p>'}
  <h2>Active baseline results</h2>
  {''.join(scan_blocks) or '<p class=dim>No scans.</p>'}
</main>
</body></html>"""
    p = out / "report.html"
    p.write_text(doc, encoding="utf-8")
    return p

# ---------------------------------------------------------------------------
# Engagement setup (interactive output location + options)
# ---------------------------------------------------------------------------

def choose_output_dir(target: str, args) -> Path:
    """Phase 0: decide where results/reports are saved.

    Priority: --out flag > interactive prompt > timestamped default.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_name = f"ayed-api-hunter-{host_key(target)}-{stamp}"
    default_dir = Path.cwd() / default_name

    if args.out:
        out = expand_path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        ok(f"Output directory (from --out): {out}")
        return out

    if not interactive_available():
        default_dir.mkdir(parents=True, exist_ok=True)
        warn(f"Non-interactive session; saving results to: {default_dir}")
        return default_dir

    print(c("\n  Where should I save the results, reports and evidence?", "bold"))
    print(c(f"  Default: {default_dir}", "dim"))
    print(c("  Enter a folder path, or press Enter to accept the default.", "dim"))
    raw = ask("Output folder", str(default_dir))
    base = expand_path(raw or str(default_dir))

    # If they pointed at an existing directory, offer to nest a run folder inside it
    # so multiple runs don't collide.
    if base.exists() and base.is_dir():
        if ask_yes_no(f"'{base.name}' exists. Create a timestamped subfolder inside it?", default=True):
            base = base / default_name

    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        warn(f"Could not create '{base}' ({e}); falling back to default.")
        base = default_dir
        base.mkdir(parents=True, exist_ok=True)

    ok(f"Results will be saved to: {base}")
    return base

def choose_formats(args) -> tuple:
    if args.formats:
        requested = tuple(x.strip().lower() for x in args.formats.split(",") if x.strip())
        invalid = [x for x in requested if x not in VALID_FORMATS]
        if invalid:
            die(f"Unknown format(s): {', '.join(invalid)}. Valid: {', '.join(VALID_FORMATS)}")
        return requested
    if not interactive_available():
        return DEFAULT_FORMATS
    print(c("\n  Which report formats do you want?", "bold"))
    print(c(f"  Available: {', '.join(VALID_FORMATS)}", "dim"))
    raw = ask("Formats (comma-separated, Enter = all)", "")
    if not raw:
        return DEFAULT_FORMATS
    requested = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    invalid = [x for x in requested if x not in VALID_FORMATS]
    if invalid:
        warn(f"Ignoring unknown format(s): {', '.join(invalid)}")
        requested = tuple(x for x in requested if x in VALID_FORMATS)
    return requested or DEFAULT_FORMATS

def engagement_setup(target: str, args):
    phase_banner(0, "ENGAGEMENT SETUP", "Authorization, scope and output location")
    print(f"  Target : {c(target, 'bold')}")
    print(f"  Scope  : strict same-host = {c(host_key(target), 'cyan')}")
    print(c("  Mode   : passive recon + low-risk HTTP checks (non-destructive)", "yellow"))

    if not args.yes and interactive_available():
        print(c(
            "\n  This tool must only be used against targets you are explicitly\n"
            "  authorized to test (your own assets or an in-scope bug bounty program).",
            "yellow"))
        if not ask_yes_no("Do you have authorization to test this target?", default=False):
            die("Authorization not confirmed. Aborting.")

    operator = args.operator or ask("Operator / handle (optional)", os.environ.get("USER", ""))
    engagement_name = args.engagement or ask("Engagement name (optional)", f"{host_key(target)}-recon")

    out = choose_output_dir(target, args)
    formats = choose_formats(args)

    engagement = {"name": engagement_name, "operator": operator}
    ok(f"Engagement '{engagement_name}' ready. Formats: {', '.join(formats)}")
    return out, formats, engagement

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Ayed Smart API Hunter (Red Team Flow) - same-host recon + low-risk curl checks.\n\n"
            "Examples:\n"
            "  python ayed_smart_api_hunter.py https://example.com\n"
            "  python ayed_smart_api_hunter.py https://example.com --out ~/loot/example --formats md,html,playbook\n"
            "  python ayed_smart_api_hunter.py https://example.com --yes --out ./run1   (non-interactive)\n"
        ),
    )
    ap.add_argument("target", help="Authorized target, e.g. https://example.com")
    ap.add_argument("--depth", type=int, default=3, help="Katana depth (default: 3)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between active curl requests")
    ap.add_argument("--max-urls", type=int, default=400, help="Maximum same-host URLs retained")
    ap.add_argument("--scan-top", type=int, default=80, help="Actively baseline top scored endpoints")
    ap.add_argument("--native-pages", type=int, default=25, help="Built-in HTML crawler page limit")
    ap.add_argument("--max-js", type=int, default=40, help="JS files inspected for endpoint strings")
    ap.add_argument("--no-archives", action="store_true", help="Skip gau and waybackurls")
    ap.add_argument("--out", help="Output directory (skips the interactive save-location prompt)")
    ap.add_argument("--formats", help=f"Comma-separated report formats ({', '.join(VALID_FORMATS)})")
    ap.add_argument("--operator", help="Operator / handle recorded in the report")
    ap.add_argument("--engagement", help="Engagement name recorded in the report")
    ap.add_argument("--yes", action="store_true", help="Non-interactive: assume authorization and defaults")
    args = ap.parse_args()

    if args.delay < MIN_DELAY: die(f"--delay must be >= {MIN_DELAY}s")
    if not 1 <= args.depth <= 8: die("--depth must be 1..8")
    if args.max_urls < 10: die("--max-urls must be >= 10")
    if args.scan_top < 1: die("--scan-top must be >= 1")

    target = normalize_target(args.target)
    if not tool_exists("curl"): die("curl required. Termux: pkg install curl")

    print(c(f"Ayed Smart API Hunter v{VERSION}", "bold"))

    # PHASE 0 — Engagement setup (asks where to save results/reports)
    out, formats, engagement = engagement_setup(target, args)

    store = DiscoveryStore(target, args.max_urls)
    store.add(target, "seed")

    # PHASE 1 — Recon (passive)
    phase_banner(1, "RECON", "robots / sitemap / katana / gau / waybackurls")
    info("Built-in robots/sitemap/HTML seed discovery")
    discover_native(target, store, args.delay, args.native_pages)

    tool_meta = []
    info("Running Katana if available")
    urls, meta = run_katana(target, args.depth); tool_meta.append(meta); store.add_many(urls, "katana")
    ok(f"Katana raw URLs: {len(urls)}") if meta["available"] else warn("Katana missing")

    if not args.no_archives:
        info("Running gau if available")
        urls, meta = run_gau(target); tool_meta.append(meta); store.add_many(urls, "gau")
        ok(f"gau raw URLs: {len(urls)}") if meta["available"] else warn("gau missing")

        info("Running waybackurls if available")
        urls, meta = run_waybackurls(target); tool_meta.append(meta); store.add_many(urls, "waybackurls")
        ok(f"waybackurls raw URLs: {len(urls)}") if meta["available"] else warn("waybackurls missing")
    else:
        tool_meta += [
            {"tool": "gau", "available": tool_exists("gau"), "command": "skipped", "stderr": ""},
            {"tool": "waybackurls", "available": tool_exists("waybackurls"), "command": "skipped", "stderr": ""},
        ]

    # PHASE 2 — Enumeration
    phase_banner(2, "ENUMERATION", "JavaScript endpoint extraction")
    info("Inspecting JavaScript for API endpoint strings")
    discover_js(target, store, args.delay, args.max_js)

    # PHASE 3 — Attack surface mapping
    phase_banner(3, "ATTACK SURFACE", "Scoring, classification and risk tiering")
    discovered = store.sorted()
    counts = tier_counts(discovered)
    ok(f"Unique same-host URLs retained: {len(discovered)}")
    for t in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
        if counts.get(t):
            print(f"    {c(t.ljust(9), TIER_COLOR[t])} {counts[t]}")

    # PHASE 4 — Active baseline
    active_candidates = [x for x in discovered if not is_static(x["url"]) and not is_js(x["url"])][:args.scan_top]
    phase_banner(4, "ACTIVE BASELINE", f"Low-risk curl checks on top {len(active_candidates)} endpoints")
    scans = []
    for idx, item in enumerate(active_candidates, 1):
        print(c(f"\n[{idx}/{len(active_candidates)}] tier={item['tier']} score={item['score']}", "magenta"))
        try:
            scans.append(scan_endpoint(item["url"], args.delay))
        except KeyboardInterrupt:
            warn("Interrupted; writing partial report")
            break
        except Exception as e:
            fail(f"{item['url']} -> {e}")

    # PHASE 5 — Analysis / playbook
    phase_banner(5, "ANALYSIS", "Prioritization and manual-test playbook")
    playbook = build_playbook(scans)
    ok(f"Playbook leads generated: {len(playbook)}")

    # PHASE 6 — Reporting
    phase_banner(6, "REPORTING", f"Writing outputs to {out}")
    data = {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "delay": args.delay,
        "engagement": engagement,
        "tool_meta": tool_meta,
        "forms": store.forms,
        "discovered": discovered,
        "scans": scans,
        "playbook": playbook,
        "tier_counts": counts,
    }
    paths = write_reports(out, data, formats)

    print()
    ok("Engagement complete")
    print(f"  Saved to: {c(str(out), 'bold')}")
    label_map = {
        "markdown": "Report (md)", "html": "Report (html)", "json": "JSON",
        "csv": "CSV", "playbook": "Playbook", "evidence": "Curl evidence",
        "urls": "All URLs", "api": "API candidates",
    }
    for key, label in label_map.items():
        if key in paths:
            print(f"  {label:<16}: {paths[key]}")

if __name__ == "__main__":
    main()
