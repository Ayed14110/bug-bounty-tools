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
  5. Active detection   - non-destructive vuln detection (detect, not exploit):
                          JWT weakness parsing, secret scanning, exposed
                          files/paths, GraphQL introspection, 403/401 bypass
                          surface, subdomain-takeover fingerprints
  6. Analysis           - flags, prioritization, manual-test playbook
  7. Reporting/handoff  - md / json / csv / html / playbook / curl evidence

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
# Active detection checks (non-destructive; detect, do NOT exploit)
#
# Boundary: every check here is a single benign request or pure local parsing.
# Nothing forges credentials, injects payloads, brute-forces, floods, or
# performs state changes. Results are LEADS for manual verification, mapped
# to techniques in RED_TEAM_METHODOLOGY.md / RED_TEAM_50_ADVANCED_PLAYS.md.
# ---------------------------------------------------------------------------

# --- JWT: decode + weakness assessment (parse only, never send forged tokens)
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*")

def _b64url_decode(seg: str) -> bytes:
    seg = seg + "=" * (-len(seg) % 4)
    import base64
    return base64.urlsafe_b64decode(seg.encode())

def assess_jwt(token: str):
    try:
        h_seg, p_seg = token.split(".")[0], token.split(".")[1]
        header = json.loads(_b64url_decode(h_seg))
        payload = json.loads(_b64url_decode(p_seg))
    except Exception:
        return None
    weaknesses = []
    alg = str(header.get("alg", "")).lower()
    if alg in ("none", ""):
        weaknesses.append("alg=none (unsigned) [T6]")
    if alg.startswith("hs"):
        weaknesses.append("HMAC alg — test weak-secret / RS256->HS256 confusion [T6]")
    if "jku" in header or "x5u" in header:
        weaknesses.append("jku/x5u header present — SSRF-to-attacker-JWKS candidate [P41]")
    if "kid" in header:
        weaknesses.append("kid header present — injection/path-traversal candidate [P42]")
    if "exp" not in payload:
        weaknesses.append("no exp claim — non-expiring token")
    sensitive = [k for k in payload if k.lower() in ("role", "admin", "is_admin", "scope", "permissions", "user_id", "uid", "email")]
    if sensitive:
        weaknesses.append("sensitive claims: " + ",".join(sensitive))
    return {"header": header, "claims": list(payload.keys()), "weaknesses": weaknesses} if weaknesses else None

def find_jwt_findings(resp: dict):
    findings = []
    seen = set()
    haystacks = []
    for vals in resp.get("headers", {}).values():
        haystacks.extend(vals)
    haystacks.append(resp.get("body_text", "")[:20000])
    for hay in haystacks:
        for tok in JWT_RE.findall(hay or ""):
            if tok in seen:
                continue
            seen.add(tok)
            a = assess_jwt(tok)
            if a:
                a["token_prefix"] = tok[:16] + "..."
                findings.append(a)
    return findings

# --- Secret scanning in already-fetched JS/HTML (pure local regex)
SECRET_PATTERNS = {
    "AWS Access Key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "AWS Secret (heuristic)": re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]"),
    "Google API Key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "Google OAuth Client": re.compile(r"\b[0-9]+-[0-9a-z]{32}\.apps\.googleusercontent\.com\b"),
    "Slack Token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    "Slack Webhook": re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+"),
    "GitHub Token": re.compile(r"\bgh[pousr]_[0-9A-Za-z]{36,}\b"),
    "GitLab Token": re.compile(r"\bglpat-[0-9A-Za-z_\-]{20,}\b"),
    "Stripe Key": re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{24,}\b"),
    "Twilio SID": re.compile(r"\bAC[0-9a-fA-F]{32}\b"),
    "SendGrid Key": re.compile(r"\bSG\.[0-9A-Za-z_\-]{22}\.[0-9A-Za-z_\-]{43}\b"),
    "Mailgun Key": re.compile(r"\bkey-[0-9a-zA-Z]{32}\b"),
    "Firebase Cloud Key": re.compile(r"\bAAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140,}\b"),
    "NPM Token": re.compile(r"\bnpm_[0-9A-Za-z]{36}\b"),
    "Private Key Block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    "Generic Bearer": re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
    "Generic Secret Assignment": re.compile(r"(?i)(?:api[_-]?key|secret|token|passwd|password)\s*[:=]\s*['\"][0-9a-zA-Z/+_\-]{16,}['\"]"),
    "JWT in source": JWT_RE,
}

# Regexes reused by passive detection checks (P81-P110)
RFC1918_RE = re.compile(r"\b(?:10\.(?:\d{1,3})\.(?:\d{1,3})\.(?:\d{1,3})|172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3})\.(?:\d{1,3})|192\.168\.(?:\d{1,3})\.(?:\d{1,3}))\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
STACKTRACE_RE = re.compile(r"(?i)(traceback \(most recent call last\)|java\.lang\.[A-Za-z.]+Exception|stack trace:|at [a-z0-9_.]+\([A-Za-z0-9_]+\.java:\d+\)|on line \d+ in|fatal error:|System\.Web\.|Microsoft\.Data\.|ORA-\d{5}|SQLSTATE\[)")
CMS_FINGERPRINTS = {
    "WordPress": re.compile(r"(?i)/wp-content/|/wp-includes/|wp-json"),
    "Drupal": re.compile(r"(?i)Drupal\.settings|/sites/default/files"),
    "Joomla": re.compile(r"(?i)/media/jui/|com_content|Joomla!"),
    "Laravel": re.compile(r"(?i)laravel_session|X-Powered-By: PHP.*Laravel"),
    "Django": re.compile(r"(?i)csrftoken|__admin__|Django"),
    "Rails": re.compile(r"(?i)csrf-param|authenticity_token|rails"),
    "Next.js": re.compile(r"(?i)__NEXT_DATA__|/_next/"),
}
SECRET_URL_PARAMS = ("password", "passwd", "pwd", "token", "access_token", "api_key",
                     "apikey", "secret", "auth", "session", "jwt", "key")

def scan_secrets(text: str, source: str):
    out = []
    for label, pat in SECRET_PATTERNS.items():
        m = pat.search(text or "")
        if m:
            snippet = m.group(0)
            out.append({"type": label, "source": source, "sample": snippet[:12] + "..." if len(snippet) > 12 else snippet, "technique": "P79/T27"})
    return out

# --- Subdomain-takeover fingerprints (match already-fetched body)
TAKEOVER_FINGERPRINTS = {
    "AWS/S3": "NoSuchBucket",
    "GitHub Pages": "There isn't a GitHub Pages site here",
    "Heroku": "No such app",
    "Shopify": "Sorry, this shop is currently unavailable",
    "Fastly": "Fastly error: unknown domain",
    "Zendesk": "Help Center Closed",
    "Surge.sh": "project not found",
    "Bitbucket": "Repository not found",
    "Unbounce": "The requested URL was not found on this server",
}

def check_takeover(body: str):
    hits = []
    low = (body or "")[:8000]
    for provider, sig in TAKEOVER_FINGERPRINTS.items():
        if sig.lower() in low.lower():
            hits.append({"provider": provider, "fingerprint": sig, "technique": "P... (subdomain takeover)"})
    return hits

# --- Exposed sensitive paths (bounded, safe GET probes)
EXPOSED_PATHS = [
    ("/.git/config", "[core]"),
    ("/.git/HEAD", "ref:"),
    ("/.env", "="),
    ("/.env.local", "="),
    ("/config.json", "{"),
    ("/actuator", "_links"),
    ("/actuator/env", "propertySources"),
    ("/server-status", "Apache Status"),
    ("/phpinfo.php", "phpinfo()"),
    ("/.DS_Store", "Bud1"),
    ("/backup.zip", "PK"),
    ("/wp-config.php.bak", "DB_PASSWORD"),
]

def check_exposed_files(origin: str, delay: float):
    findings = []
    for path, sig in EXPOSED_PATHS:
        time.sleep(delay)
        resp = curl_request(origin + path, "GET")
        code = int(resp.get("meta", {}).get("http_code") or 0)
        body = resp.get("body_text", "")
        if code == 200 and (sig.lower() in body.lower() or (sig == "PK" and body[:2] == "PK")):
            findings.append({"url": origin + path, "status": code, "signature": sig,
                             "size": resp.get("meta", {}).get("size_download"),
                             "technique": "T27/P72", "command": resp.get("command")})
    return findings

# --- GraphQL introspection (single benign query)
def check_graphql_introspection(url: str, delay: float):
    time.sleep(delay)
    query = '{"query":"query{__schema{queryType{name}}}"}'
    with tempfile.TemporaryDirectory(prefix="ayed_gql_") as td:
        bfile = Path(td) / "b.json"
        cmd = ["curl", "-sS", "--max-time", str(DEFAULT_TIMEOUT), "-A", UA,
               "-H", "Content-Type: application/json", "-X", "POST",
               "-o", str(bfile), "-w", "%{http_code}", "--data", query, url]
        p = subprocess.run(cmd, capture_output=True, text=True)
        body = bfile.read_text(errors="replace") if bfile.exists() else ""
    enabled = "__schema" in body and "queryType" in body
    return {
        "url": url, "introspection_enabled": enabled,
        "status": p.stdout.strip(), "technique": "T22/P31",
        "command": " ".join(shlex.quote(x) for x in cmd),
    }

# --- 403/401 bypass surface probe (bounded, GET-only, non-destructive)
BYPASS_VARIANTS = [
    ("path", "/%2e/"), ("path", "/./"), ("path", "//"), ("path", "/..;/"),
    ("header", "X-Original-URL"), ("header", "X-Rewrite-URL"),
    ("header", "X-Forwarded-For"), ("header", "X-Forwarded-Host"),
]

def check_403_bypass(url: str, baseline_code: int, delay: float):
    p = urlsplit(url)
    attempts = []
    for kind, token in BYPASS_VARIANTS:
        time.sleep(delay)
        if kind == "path":
            test_url = urlunsplit((p.scheme, p.netloc, (p.path.rstrip("/") + token), p.query, ""))
            resp = curl_request(test_url, "GET")
        else:
            test_url = url
            hv = "127.0.0.1" if "For" in token or "Host" in token else p.path
            resp = curl_request(url, "GET", headers=[f"{token}: {hv}"])
        code = int(resp.get("meta", {}).get("http_code") or 0)
        if code and code != baseline_code and code < 400:
            attempts.append({"variant": f"{kind}:{token}", "status": code,
                             "baseline": baseline_code, "technique": "T5",
                             "command": resp.get("command")})
    return attempts

# --- Passive per-response detection (local parse; NO extra requests) P81-P96
def passive_scan_findings(scan: dict):
    base = scan.get("tests", {}).get("get", {})
    body = base.get("body_text", "")[:60000]
    headers = base.get("headers", {})
    ct = (first_header(base, "content-type") or "").lower()
    sh = scan.get("security_headers", {})
    out = []

    def add(kind, detail, technique):
        out.append({"type": kind, "detail": detail, "technique": technique})

    # P81 verbose error / stack trace disclosure
    m = STACKTRACE_RE.search(body)
    if m:
        add("verbose_error", f"stack-trace/error signature: {m.group(0)[:60]}", "P81")
    # P82 technology / version disclosure
    tech = []
    for hn in ("server", "x-powered-by", "x-aspnet-version", "x-aspnetmvc-version", "x-generator"):
        v = first_header(base, hn)
        if v:
            tech.append(f"{hn}: {v}")
    if tech:
        add("tech_disclosure", "; ".join(tech), "P82")
    # P83 directory listing
    if "index of /" in body.lower() or "<title>index of" in body.lower():
        add("directory_listing", "Apache/nginx autoindex page", "P83")
    # P84 internal IP disclosure
    ips = set(RFC1918_RE.findall(body))
    for vals in headers.values():
        for v in vals:
            ips.update(RFC1918_RE.findall(v))
    if ips:
        add("internal_ip", "RFC1918 addresses: " + ",".join(sorted(ips)[:5]), "P84")
    # P85 email / PII exposure
    emails = set(EMAIL_RE.findall(body))
    if len(emails) >= 3:
        add("pii_email", f"{len(emails)} email addresses in response", "P85")
    # P86 clickjacking (missing frame protections)
    csp = (first_header(base, "content-security-policy") or "").lower()
    if not first_header(base, "x-frame-options") and "frame-ancestors" not in csp:
        add("clickjacking", "no X-Frame-Options and no CSP frame-ancestors", "P86")
    # P87 content-type sniffing risk
    if "X-Content-Type-Options" in sh.get("missing", []):
        add("content_sniffing", "missing X-Content-Type-Options: nosniff", "P87")
    # P88 cacheable sensitive JSON (web cache deception indicator)
    if "json" in ct:
        cc = (first_header(base, "cache-control") or "").lower()
        if not cc or ("no-store" not in cc and "no-cache" not in cc and "private" not in cc):
            add("cacheable_json", f"JSON without no-store/private (cache-control: {cc or 'none'})", "P88")
    # P89 missing rate-limit headers on auth/sensitive endpoint
    if set(scan.get("tags", [])) & {"auth", "identity", "api"}:
        if not any(first_header(base, h) for h in ("x-ratelimit-limit", "ratelimit-limit", "retry-after", "x-rate-limit-limit")):
            add("no_ratelimit_headers", "no rate-limit headers advertised on sensitive endpoint", "P89")
    # P90 secret/token in URL query string
    low_url = scan["url"].lower()
    hit = [p for p in SECRET_URL_PARAMS if re.search(rf"[?&]{re.escape(p)}=", low_url)]
    if hit or JWT_RE.search(scan["url"]):
        add("secret_in_url", "sensitive params in URL: " + (",".join(hit) or "jwt"), "P90")
    # P91 mixed content on https page
    if scan["url"].startswith("https://") and "html" in ct:
        if re.search(r'(?:src|href|action)\s*=\s*["\']http://', body):
            add("mixed_content", "http:// resource referenced from https page", "P91")
    # P92 CMS / framework fingerprint
    hay = body + " " + " ".join(f"{k}:{','.join(v)}" for k, v in headers.items())
    for name, pat in CMS_FINGERPRINTS.items():
        if pat.search(hay):
            add("cms_fingerprint", name, "P92")
            break
    # P93 weak HSTS (present but no includeSubDomains/preload)
    hsts = first_header(base, "strict-transport-security")
    if hsts and ("includesubdomains" not in hsts.lower()):
        add("weak_hsts", f"HSTS without includeSubDomains: {hsts[:60]}", "P93")
    # P94 SameSite=None cookie without Secure / missing SameSite
    for ck in scan.get("cookies", []):
        if not ck.get("samesite"):
            add("cookie_samesite", f"cookie '{ck['name']}' missing SameSite", "P94")
    # P95 GraphQL/verbose debug keywords
    if re.search(r"(?i)\b(debug\s*=\s*true|APP_DEBUG|whoops|werkzeug|phpinfo\(\))", body):
        add("debug_mode", "debug/verbose framework signature in body", "P95")
    # P96 open CORS on OPTIONS advertising credentials
    opt = scan.get("tests", {}).get("options", {})
    if (first_header(opt, "access-control-allow-credentials") or "").lower() == "true" and \
       first_header(opt, "access-control-allow-origin") == "*":
        add("cors_wildcard_creds", "OPTIONS advertises ACAO:* with credentials", "P96")
    return out

# --- Host-level active detection (bounded extra requests) P97-P110
def host_level_checks(target: str, store, delay: float):
    origin = base_origin(target)
    host = host_key(target)
    findings = {}

    # P97 security.txt
    time.sleep(delay)
    r = curl_request(origin + "/.well-known/security.txt", "GET")
    if int(r.get("meta", {}).get("http_code") or 0) == 200 and "contact" in r.get("body_text", "").lower():
        findings["security_txt"] = {"url": origin + "/.well-known/security.txt", "technique": "P97"}

    # P98 OIDC discovery
    time.sleep(delay)
    r = curl_request(origin + "/.well-known/openid-configuration", "GET")
    if int(r.get("meta", {}).get("http_code") or 0) == 200 and "authorization_endpoint" in r.get("body_text", ""):
        try:
            cfg = json.loads(r.get("body_text", "") or "{}")
            findings["oidc"] = {"url": origin + "/.well-known/openid-configuration",
                                "authorization_endpoint": cfg.get("authorization_endpoint"),
                                "token_endpoint": cfg.get("token_endpoint"), "technique": "P98"}
        except Exception:
            pass

    # P99 CORS null-origin acceptance
    time.sleep(delay)
    r = curl_request(target, "GET", headers=["Origin: null"])
    if first_header(r, "access-control-allow-origin") == "null":
        findings["cors_null"] = {"url": target, "detail": "ACAO reflects 'null' origin", "technique": "P99"}

    # P100 CORS naive matching (subdomain / suffix trust)
    for probe in (f"https://ayedbbp.{host}", f"https://not{host}"):
        time.sleep(delay)
        r = curl_request(target, "GET", headers=[f"Origin: {probe}"])
        if first_header(r, "access-control-allow-origin") == probe:
            findings["cors_naive_match"] = {"url": target, "reflected_origin": probe, "technique": "P100"}
            break

    # P101 TRACE method (XST)
    time.sleep(delay)
    r = curl_request(target, "TRACE")
    if int(r.get("meta", {}).get("http_code") or 0) == 200 and "trace" in (r.get("body_text", "")[:200].lower()):
        findings["trace_method"] = {"url": target, "technique": "P101"}

    # P102 Host header reflection
    time.sleep(delay)
    marker = "ayed-hbp-" + uuid.uuid4().hex[:8] + ".invalid"
    r = curl_request(target, "GET", headers=[f"X-Forwarded-Host: {marker}"])
    loc = first_header(r, "location") or ""
    if marker in r.get("body_text", "")[:60000] or marker in loc:
        findings["host_header_reflection"] = {"url": target, "marker": marker, "technique": "P102"}

    # P103 WordPress REST user enumeration
    time.sleep(delay)
    r = curl_request(origin + "/wp-json/wp/v2/users", "GET")
    if int(r.get("meta", {}).get("http_code") or 0) == 200 and '"slug"' in r.get("body_text", ""):
        findings["wp_user_enum"] = {"url": origin + "/wp-json/wp/v2/users", "technique": "P103"}

    # P104 common admin/debug panels present
    panels = []
    for path in ("/admin/", "/administrator/", "/manage/", "/dashboard/", "/debug/", "/status", "/metrics"):
        time.sleep(delay)
        r = curl_request(origin + path, "GET")
        code = int(r.get("meta", {}).get("http_code") or 0)
        if code in (200, 401, 403):
            panels.append({"path": path, "status": code})
    if panels:
        findings["admin_panels"] = {"panels": panels, "technique": "P104"}

    return findings

# --- JS source-map exposure (bounded) P105
def check_source_maps(store, delay: float, limit: int = 10):
    js_urls = [u for u in store.urls if is_js(u)][:limit]
    out = []
    for js in js_urls:
        time.sleep(delay)
        r = curl_request(js + ".map", "GET", max_body=MAX_JS_BODY)
        if int(r.get("meta", {}).get("http_code") or 0) == 200 and '"mappings"' in r.get("body_text", "")[:5000]:
            out.append({"url": js + ".map", "technique": "P105"})
    return out

# --- Backup / temp file exposure for discovered paths (bounded) P106
def check_backup_files(store, delay: float, limit: int = 8):
    candidates = [u for u in store.sorted() if not is_static(u["url"]) and not is_js(u["url"])
                  and urlsplit(u["url"]).path not in ("", "/")][:limit]
    out = []
    for item in candidates:
        for suffix in (".bak", "~"):
            time.sleep(delay)
            r = curl_request(item["url"] + suffix, "GET")
            if int(r.get("meta", {}).get("http_code") or 0) == 200 and int(r.get("meta", {}).get("size_download") or 0) > 0:
                out.append({"url": item["url"] + suffix, "technique": "P106"})
                break
    return out

# =========================================================================
# Advanced detection engine (data-driven) — 100+ real checks
#
# Sources of checks:
#   - BODY_SIGNATURES  : ~35 response-body signatures (SQL errors per DBMS,
#                        stack traces per stack, debug pages, info leaks)
#   - HEADER_RULES     : ~12 header-based detections
#   - TECH_FINGERPRINTS: ~30 technology / framework / WAF / CDN fingerprints
#   - SENSITIVE_PATHS  : ~45 sensitive path/file exposures (bounded GET probes)
#   - active_probes()  : benign marker probes for reflection / SQL-error /
#                        SSTI / LFI classes — OPT-IN via --intrusive only.
#
# Everything DETECTS; nothing weaponizes. No brute force, no flooding, no
# request smuggling, no forged-auth replay, no internal SSRF, no data
# destruction. Intrusive probes send a single benign marker per parameter.
# =========================================================================

def _rx(p):
    return re.compile(p, re.I)

# (id, title, category, severity, technique, regex)
BODY_SIGNATURES = [
    # --- SQL errors by DBMS
    ("sql-mysql", "MySQL error", "sql-error", "high", "T14/P56", _rx(r"you have an error in your sql syntax|warning:\s*mysql_|mysql_fetch|com\.mysql\.jdbc|valid MySQL result")),
    ("sql-postgres", "PostgreSQL error", "sql-error", "high", "T14/P56", _rx(r"pg_query\(\)|PostgreSQL.*ERROR|org\.postgresql\.util\.PSQLException|unterminated quoted string")),
    ("sql-mssql", "MSSQL error", "sql-error", "high", "T14/P56", _rx(r"unclosed quotation mark|microsoft ole db provider|odbc sql server driver|System\.Data\.SqlClient|Incorrect syntax near")),
    ("sql-oracle", "Oracle error", "sql-error", "high", "T14/P56", _rx(r"ORA-\d{5}|quoted string not properly terminated|oracle\.jdbc")),
    ("sql-sqlite", "SQLite error", "sql-error", "high", "T14/P56", _rx(r"sqlite3?\.OperationalError|SQLite/JDBCDriver|sqlite_error")),
    ("sql-generic", "Generic SQL error", "sql-error", "high", "T14/P56", _rx(r"SQLSTATE\[|sql syntax.*error|unexpected end of sql")),
    # --- Stack traces by stack
    ("st-php", "PHP error/stack", "stack-trace", "medium", "P81", _rx(r"(fatal error|warning|notice|parse error):.*(on line|in .*\.php)|Stack trace:\s*#0")),
    ("st-aspnet", "ASP.NET error", "stack-trace", "medium", "P81", _rx(r"Server Error in .* Application|System\.Web\.|Microsoft \.NET Framework|\[SqlException")),
    ("st-java", "Java/Spring trace", "stack-trace", "medium", "P81", _rx(r"javax?\.servlet|org\.springframework|java\.lang\.[A-Za-z.]+Exception|at [\w.$]+\([\w]+\.java:\d+\)")),
    ("st-python", "Python traceback", "stack-trace", "medium", "P81", _rx(r"Traceback \(most recent call last\)|Werkzeug Debugger|Django Version|File \".*\.py\", line \d+")),
    ("st-ruby", "Ruby/Rails trace", "stack-trace", "medium", "P81", _rx(r"ActionController::|app/controllers/|/gems/|rails\.error")),
    ("st-node", "Node.js trace", "stack-trace", "medium", "P81", _rx(r"at Object\.<anonymous>|/node_modules/|ReferenceError:.*at ")),
    # --- Debug / dangerous pages
    ("dbg-phpinfo", "phpinfo() page", "debug", "high", "P95", _rx(r"<title>phpinfo\(\)|PHP Version</td>")),
    ("dbg-whoops", "Whoops debug page", "debug", "high", "P95", _rx(r"Whoops, looks like something went wrong|/vendor/filp/whoops")),
    ("dbg-werkzeug", "Werkzeug console", "debug", "high", "P95", _rx(r"Werkzeug Debugger|__debugger__|console-lock")),
    ("dbg-django", "Django DEBUG page", "debug", "high", "P95", _rx(r"You're seeing this error because you have <code>DEBUG = True|Django tried these URL patterns")),
    ("dbg-railsdev", "Rails dev error", "debug", "high", "P95", _rx(r"Full Trace</a>|Application Trace</a>|Rails\.root:")),
    # --- Info leaks
    ("leak-abs-path-unix", "Unix path disclosure", "info-leak", "low", "P84", _rx(r"/(?:var/www|home/[\w.-]+|usr/local|opt)/[\w./-]+")),
    ("leak-abs-path-win", "Windows path disclosure", "info-leak", "low", "P84", _rx(r"[C-Z]:\\\\(?:inetpub|windows|users|xampp|wwwroot)")),
    ("leak-aws-arn", "AWS ARN", "info-leak", "medium", "P71", _rx(r"arn:aws:[a-z0-9-]+:[a-z0-9-]*:\d{12}:")),
    ("leak-s3-url", "S3 bucket URL", "info-leak", "medium", "P71", _rx(r"[a-z0-9.-]+\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com")),
    ("leak-gcp-sa", "GCP service account", "info-leak", "high", "P75", _rx(r"\"type\"\s*:\s*\"service_account\"")),
    ("leak-private-key", "Private key block", "info-leak", "critical", "P79", _rx(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("leak-conn-string", "DB connection string", "info-leak", "high", "P79", _rx(r"(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis)://[^\s\"'<>]+:[^\s\"'<>]+@")),
    ("leak-jwt", "JWT token", "info-leak", "medium", "T6", JWT_RE),
]

# (id, title, technique, header, regex-or-None) — regex None means "present"
HEADER_RULES = [
    ("hdr-debug", "Debug header exposed", "P95", "x-debug", None),
    ("hdr-runtime", "Runtime version leak", "P82", "x-runtime", None),
    ("hdr-powered", "X-Powered-By disclosure", "P82", "x-powered-by", None),
    ("hdr-aspnet", "ASP.NET version leak", "P82", "x-aspnet-version", None),
    ("hdr-backend", "Backend server header", "P82", "x-backend-server", None),
    ("hdr-cache-hit", "Cache hit header (poisoning surface)", "P63", "x-cache", _rx(r"hit")),
    ("hdr-via-proxy", "Via/proxy header", "P67", "via", None),
    ("hdr-amz-id", "AWS request id header", "P71", "x-amz-request-id", None),
    ("hdr-cf-ray", "Cloudflare ray id", "waf", "cf-ray", None),
    ("hdr-set-cookie-debug", "Debug cookie", "P95", "set-cookie", _rx(r"(?:debug|test|dev)=")),
]

# name -> (kind, regex) over combined body+headers
TECH_FINGERPRINTS = {
    "WordPress": ("cms", _rx(r"/wp-content/|/wp-includes/|wp-json|X-Pingback")),
    "Drupal": ("cms", _rx(r"Drupal\.settings|/sites/default/files|X-Drupal-Cache|X-Generator: Drupal")),
    "Joomla": ("cms", _rx(r"/media/jui/|com_content|Joomla!|/administrator/")),
    "Magento": ("cms", _rx(r"/skin/frontend/|Mage\.Cookies|/static/version\d+")),
    "Shopify": ("cms", _rx(r"cdn\.shopify\.com|Shopify\.theme|x-shopid")),
    "Laravel": ("framework", _rx(r"laravel_session|XSRF-TOKEN|/vendor/laravel")),
    "Django": ("framework", _rx(r"csrftoken|__admin_media_prefix__|X-Frame-Options: DENY.*wsgi")),
    "Flask/Werkzeug": ("framework", _rx(r"Werkzeug/|Server: Werkzeug")),
    "Ruby on Rails": ("framework", _rx(r"csrf-param|authenticity_token|X-Runtime|_rails")),
    "Express": ("framework", _rx(r"X-Powered-By: Express")),
    "Next.js": ("framework", _rx(r"__NEXT_DATA__|/_next/static|x-nextjs")),
    "Nuxt/Vue": ("framework", _rx(r"__NUXT__|data-n-head|/_nuxt/")),
    "Angular": ("framework", _rx(r"ng-version|/@angular/|ng-app")),
    "React": ("framework", _rx(r"data-reactroot|__REACT_DEVTOOLS|react-dom")),
    "Spring Boot": ("framework", _rx(r"whitelabel error page|/actuator|org\.springframework\.boot")),
    "ASP.NET": ("framework", _rx(r"__VIEWSTATE|ASP\.NET|X-AspNet-Version")),
    "Nginx": ("server", _rx(r"Server:\s*nginx")),
    "Apache": ("server", _rx(r"Server:\s*Apache")),
    "IIS": ("server", _rx(r"Server:\s*Microsoft-IIS")),
    "OpenResty": ("server", _rx(r"Server:\s*openresty")),
    "Cloudflare": ("cdn/waf", _rx(r"Server:\s*cloudflare|cf-ray|__cf_bm")),
    "Akamai": ("cdn/waf", _rx(r"AkamaiGHost|X-Akamai|akamai")),
    "Fastly": ("cdn/waf", _rx(r"X-Served-By:\s*cache.*fastly|Fastly-")),
    "Imperva/Incapsula": ("waf", _rx(r"incap_ses|visid_incap|X-Iinfo|Incapsula")),
    "F5 BIG-IP": ("waf", _rx(r"BIGipServer|TS[0-9a-f]{8}=")),
    "Sucuri": ("waf", _rx(r"X-Sucuri-ID|Sucuri/Cloudproxy")),
    "AWS WAF/ALB": ("cdn/waf", _rx(r"awselb=|AWSALB|x-amzn-|Server:\s*awselb")),
    "ModSecurity": ("waf", _rx(r"Mod_Security|NOYB|This error was generated by Mod_Security")),
}

# (path, body-signature-substring or None, severity, technique)
SENSITIVE_PATHS = [
    ("/.git/config", "[core]", "high", "T27/P72"),
    ("/.git/HEAD", "ref:", "high", "T27/P72"),
    ("/.gitignore", None, "info", "T27"),
    ("/.svn/entries", None, "medium", "T27"),
    ("/.hg/requires", None, "medium", "T27"),
    ("/.env", "=", "critical", "P79"),
    ("/.env.local", "=", "critical", "P79"),
    ("/.env.production", "=", "critical", "P79"),
    ("/config.json", "{", "medium", "P79"),
    ("/config.yml", ":", "medium", "P79"),
    ("/config.php.bak", "<?php", "high", "P106"),
    ("/wp-config.php.bak", "DB_PASSWORD", "critical", "P106"),
    ("/settings.py", "SECRET_KEY", "critical", "P79"),
    ("/appsettings.json", "ConnectionStrings", "high", "P79"),
    ("/web.config", "<configuration", "medium", "P79"),
    ("/docker-compose.yml", "services:", "medium", "P72"),
    ("/Dockerfile", "FROM ", "low", "P72"),
    ("/.dockerenv", None, "low", "P72"),
    ("/.aws/credentials", "aws_access_key_id", "critical", "P71"),
    ("/.npmrc", "_authToken", "high", "P72"),
    ("/.htpasswd", ":", "high", "P72"),
    ("/.DS_Store", None, "low", "T27"),
    ("/backup.zip", None, "medium", "P106"),
    ("/backup.sql", None, "high", "P106"),
    ("/dump.sql", None, "high", "P106"),
    ("/database.sql", None, "high", "P106"),
    ("/phpinfo.php", "phpinfo()", "high", "P95"),
    ("/info.php", "phpinfo()", "high", "P95"),
    ("/server-status", "Apache Server Status", "medium", "P82"),
    ("/server-info", "Apache Server Information", "medium", "P82"),
    ("/actuator", "_links", "high", "P95"),
    ("/actuator/env", "propertySources", "critical", "P95"),
    ("/actuator/health", "status", "low", "P95"),
    ("/actuator/heapdump", None, "critical", "P95"),
    ("/actuator/mappings", "dispatcherServlet", "medium", "P95"),
    ("/metrics", "# HELP", "low", "P82"),
    ("/debug/vars", "cmdline", "medium", "P95"),
    ("/.well-known/security.txt", "Contact", "info", "P97"),
    ("/robots.txt", None, "info", "recon"),
    ("/crossdomain.xml", "cross-domain", "low", "T28"),
    ("/graphql", None, "medium", "T22"),
    ("/swagger.json", "swagger", "medium", "T23"),
    ("/openapi.json", "openapi", "medium", "T23"),
    ("/api/swagger.json", "swagger", "medium", "T23"),
    ("/.well-known/openid-configuration", "authorization_endpoint", "info", "P98"),
]

def scan_body_signatures(scan: dict):
    base = scan.get("tests", {}).get("get", {})
    body = base.get("body_text", "")[:120000]
    header_blob = " ".join(f"{k}: {v}" for k, vals in base.get("headers", {}).items() for v in vals)
    hay = body + "\n" + header_blob
    out = []
    for sid, title, cat, sev, tech, rx in BODY_SIGNATURES:
        m = rx.search(hay)
        if m:
            out.append({"id": sid, "title": title, "category": cat, "severity": sev,
                        "technique": tech, "match": m.group(0)[:80]})
    return out

def scan_header_rules(scan: dict):
    base = scan.get("tests", {}).get("get", {})
    out = []
    for hid, title, tech, header, rx in HEADER_RULES:
        vals = base.get("headers", {}).get(header, [])
        for v in vals:
            if rx is None or rx.search(v):
                out.append({"id": hid, "title": title, "technique": tech,
                            "header": header, "value": v[:80]})
                break
    return out

def fingerprint_tech(scan: dict):
    base = scan.get("tests", {}).get("get", {})
    body = base.get("body_text", "")[:60000]
    header_blob = "\n".join(f"{k}: {v}" for k, vals in base.get("headers", {}).items() for v in vals)
    hay = header_blob + "\n" + body
    out = []
    for name, (kind, rx) in TECH_FINGERPRINTS.items():
        if rx.search(hay):
            out.append({"name": name, "kind": kind})
    return out

def check_sensitive_paths(origin: str, delay: float, limit: int):
    findings = []
    for path, sig, sev, tech in SENSITIVE_PATHS[:limit]:
        time.sleep(delay)
        r = curl_request(origin + path, "GET")
        code = int(r.get("meta", {}).get("http_code") or 0)
        body = r.get("body_text", "")
        size = int(r.get("meta", {}).get("size_download") or 0)
        if code == 200 and size > 0 and (sig is None or sig.lower() in body.lower()):
            findings.append({"url": origin + path, "status": code, "severity": sev,
                             "technique": tech, "size": size, "command": r.get("command")})
    return findings

# --- Intrusive active probes (OPT-IN via --intrusive): benign markers only
_SQL_ERR_RX = _rx(r"you have an error in your sql syntax|unclosed quotation mark|ORA-\d{5}|PostgreSQL.*ERROR|sqlite3?\.OperationalError|SQLSTATE\[|Incorrect syntax near|unterminated quoted string")

_LDAP_ERR_RX = _rx(r"LDAP: error|Invalid DN syntax|javax\.naming\.|com\.sun\.jndi|ldap_search|Bad search filter")
_XPATH_ERR_RX = _rx(r"XPathException|xmlXPathEval|SimpleXMLElement::xpath|Invalid expression|MS\.Internal\.Xml|XPath error")
_NOSQL_ERR_RX = _rx(r"MongoError|CastError|BSONError|unexpected token.*mongo|\$where|E11000|couldn't parse json operator")

# ---- 40 active detection methods (benign payloads; detect, do not exploit) ----
# Each: id, class, technique, context, payload(orig,marker)->value, detector, raw?
# detector kinds: reflect | eval | sql_error | ldap_error | xpath_error |
#                 nosql_error | location | read_canary | header_reflect
def _pd(id_, cls, tech, ctx, payload, kind, expect=None, raw=False):
    return {"id": id_, "cls": cls, "tech": tech, "ctx": ctx, "payload": payload,
            "kind": kind, "expect": expect, "raw": raw}

ACTIVE_PROBE_DEFS = [
    # --- XSS reflection contexts (T11) — detect unencoded reflection
    _pd("xss-html", "XSS", "T11", "html body", lambda o, m: f"{m}<b>x</b>", "reflect", lambda m: f"{m}<b>x</b>"),
    _pd("xss-attr", "XSS", "T11", "attribute", lambda o, m: f'{m}"><svg', "reflect", lambda m: f'{m}"><svg'),
    _pd("xss-js", "XSS", "T11", "js string", lambda o, m: f"{m}';x//", "reflect", lambda m: f"{m}';x//"),
    _pd("xss-svg", "XSS", "T11", "svg/markup", lambda o, m: f"{m}<svg/onload=1>", "reflect", lambda m: f"{m}<svg/onload=1>"),
    _pd("xss-url", "XSS", "T11", "url/href", lambda o, m: f"javascript:{m}//", "reflect", lambda m: f"javascript:{m}//"),
    _pd("xss-angular", "XSS", "T11", "template inj", lambda o, m: f"{m}{{{{1}}}}", "reflect", lambda m: f"{m}{{{{1}}}}"),
    _pd("xss-onerror", "XSS", "T11", "img onerror", lambda o, m: f'{m}"><img src=x onerror=1>', "reflect", lambda m: f'{m}"><img src=x onerror=1>'),
    _pd("xss-close-script", "XSS", "T11", "script close", lambda o, m: f"{m}</script>", "reflect", lambda m: f"{m}</script>"),
    # --- SQL injection error-based (T14), several syntaxes
    _pd("sqli-quote", "SQLI_ERROR", "T14", "single quote", lambda o, m: o + "'", "sql_error"),
    _pd("sqli-dquote", "SQLI_ERROR", "T14", "double quote", lambda o, m: o + '"', "sql_error"),
    _pd("sqli-paren", "SQLI_ERROR", "T14", "paren break", lambda o, m: o + "')", "sql_error"),
    _pd("sqli-comment", "SQLI_ERROR", "T14", "comment", lambda o, m: o + "'-- -", "sql_error"),
    _pd("sqli-backtick", "SQLI_ERROR", "T14", "backtick", lambda o, m: o + "`", "sql_error"),
    _pd("sqli-hash", "SQLI_ERROR", "T14", "hash comment", lambda o, m: o + "'#", "sql_error"),
    # --- NoSQL injection error (T14/P56)
    _pd("nosql-obj", "NOSQL_ERROR", "T14", "operator break", lambda o, m: o + '"}]', "nosql_error"),
    _pd("nosql-where", "NOSQL_ERROR", "T14", "$where", lambda o, m: o + "'||'1'=='1", "nosql_error"),
    # --- SSTI across engines (T13) — detect evaluated 49
    _pd("ssti-jinja", "SSTI", "T13", "Jinja/Twig", lambda o, m: f"{m}{{{{7*7}}}}", "eval", lambda m: f"{m}49"),
    _pd("ssti-dollar", "SSTI", "T13", "JSP/Spring EL", lambda o, m: f"{m}${{7*7}}", "eval", lambda m: f"{m}49"),
    _pd("ssti-hash", "SSTI", "T13", "Ruby/Slim", lambda o, m: f"{m}#{{7*7}}", "eval", lambda m: f"{m}49"),
    _pd("ssti-erb", "SSTI", "T13", "ERB", lambda o, m: f"{m}<%= 7*7 %>", "eval", lambda m: f"{m}49"),
    _pd("ssti-freemarker", "SSTI", "T13", "Freemarker", lambda o, m: f"{m}${{7*7}}", "eval", lambda m: f"{m}49"),
    _pd("ssti-velocity", "SSTI", "T13", "Velocity", lambda o, m: f"{m}#set($x=7*7)$x", "eval", lambda m: f"{m}49"),
    _pd("ssti-smarty", "SSTI", "T13", "Smarty", lambda o, m: f"{m}{{7*7}}", "eval", lambda m: f"{m}49"),
    # --- LFI / path traversal read canaries (T12) — proof only, content not stored
    _pd("lfi-unix", "LFI_TRAVERSAL", "T12", "unix passwd", lambda o, m: "../../../../../../etc/passwd", "read_canary", lambda m: "root:x:0:0"),
    _pd("lfi-unix-enc", "LFI_TRAVERSAL", "T12", "encoded", lambda o, m: "..%2f..%2f..%2f..%2f..%2fetc%2fpasswd", "read_canary", lambda m: "root:x:0:0"),
    _pd("lfi-unix-dbl", "LFI_TRAVERSAL", "T12", "nested", lambda o, m: "....//....//....//....//etc/passwd", "read_canary", lambda m: "root:x:0:0"),
    _pd("lfi-win", "LFI_TRAVERSAL", "T12", "windows ini", lambda o, m: "..\\..\\..\\..\\..\\..\\windows\\win.ini", "read_canary", lambda m: "[extensions]"),
    _pd("lfi-proc", "LFI_TRAVERSAL", "T12", "proc self", lambda o, m: "../../../../../../proc/self/environ", "read_canary", lambda m: "PATH="),
    # --- Open redirect variants (T21) — detect external Location
    _pd("redir-slashes", "OPEN_REDIRECT", "T21", "//host", lambda o, m: "//example.invalid/ayed", "location", lambda m: "example.invalid"),
    _pd("redir-backslash", "OPEN_REDIRECT", "T21", "/\\host", lambda o, m: "/\\example.invalid/ayed", "location", lambda m: "example.invalid"),
    _pd("redir-abs", "OPEN_REDIRECT", "T21", "absolute", lambda o, m: "https://example.invalid/ayed", "location", lambda m: "example.invalid"),
    _pd("redir-triple", "OPEN_REDIRECT", "T21", "///host", lambda o, m: "///example.invalid/ayed", "location", lambda m: "example.invalid"),
    _pd("redir-at", "OPEN_REDIRECT", "T21", "@host", lambda o, m: "https://trusted@example.invalid/", "location", lambda m: "example.invalid"),
    _pd("redir-space", "OPEN_REDIRECT", "T21", "whitespace", lambda o, m: " https://example.invalid/ayed", "location", lambda m: "example.invalid"),
    # --- LDAP injection error (T14)
    _pd("ldap-star", "LDAP_INJ", "T14", "filter break", lambda o, m: o + "*)(uid=*", "ldap_error"),
    _pd("ldap-amp", "LDAP_INJ", "T14", "and break", lambda o, m: o + "*)(&", "ldap_error"),
    # --- XPath injection error (T14)
    _pd("xpath-or", "XPATH_INJ", "T14", "or true", lambda o, m: o + "' or '1'='1", "xpath_error"),
    _pd("xpath-bracket", "XPATH_INJ", "T14", "bracket", lambda o, m: o + "']", "xpath_error"),
    # --- CRLF / header injection (T?) — detect injected header reflected
    _pd("crlf-header", "CRLF_INJECTION", "P54", "response split", lambda o, m: f"{o}%0d%0aX-Ayed-CRLF:{m}", "header_reflect", lambda m: m, True),
    _pd("crlf-cookie", "CRLF_INJECTION", "P54", "set-cookie", lambda o, m: f"{o}%0d%0aSet-Cookie:ayed={m}", "header_reflect", lambda m: m, True),
]

def _detect_probe(kind, resp, expect, base_body, base_has_sql):
    body = resp.get("body_text", "")
    if kind == "reflect":
        return expect in body
    if kind == "eval":
        return expect in body
    if kind == "read_canary":
        return expect in body
    if kind == "sql_error":
        return bool(_SQL_ERR_RX.search(body)) and not base_has_sql
    if kind == "ldap_error":
        return bool(_LDAP_ERR_RX.search(body))
    if kind == "xpath_error":
        return bool(_XPATH_ERR_RX.search(body))
    if kind == "nosql_error":
        return bool(_NOSQL_ERR_RX.search(body))
    if kind == "location":
        loc = first_header(resp, "location") or ""
        return expect in loc
    if kind == "header_reflect":
        for k, vals in resp.get("headers", {}).items():
            for v in vals:
                if expect in v:
                    return True
        return False
    return False

def active_probes(scan: dict, delay: float, max_params: int = 3, budget=None):
    """Run the 40-method active detection library against a URL's parameters.

    Sends one benign payload per (parameter, method) and DETECTS the class.
    Respects a shared request budget (list [remaining]) to avoid flooding.
    """
    url = scan["url"]
    params = parse_qsl(urlsplit(url).query, keep_blank_values=True)
    if not params:
        return []
    base_body = scan.get("tests", {}).get("get", {}).get("body_text", "")
    base_has_sql = bool(_SQL_ERR_RX.search(base_body))
    p = urlsplit(url)
    results = []

    def send(target_param_idx, value, raw):
        if raw:
            parts = []
            for j, (k, v) in enumerate(params):
                parts.append(f"{k}={value if j == target_param_idx else v}")
            qs = "&".join(parts)
        else:
            ch = list(params); ch[target_param_idx] = (params[target_param_idx][0], value)
            qs = urlencode(ch, doseq=True)
        test_url = urlunsplit((p.scheme, p.netloc, p.path, qs, ""))
        time.sleep(delay)
        return test_url, curl_request(test_url, "GET")

    for idx, (key, original) in enumerate(params[:max_params]):
        for d in ACTIVE_PROBE_DEFS:
            if budget is not None:
                if budget[0] <= 0:
                    return results
                budget[0] -= 1
            marker = "ayed" + uuid.uuid4().hex[:8]
            value = d["payload"](original, marker)
            expect = d["expect"](marker) if d["expect"] else None
            turl, r = send(idx, value, d["raw"])
            if _detect_probe(d["kind"], r, expect, base_body, base_has_sql):
                results.append({"parameter": key, "class": d["cls"], "method": d["id"],
                                "context": d["ctx"], "technique": d["tech"],
                                "detail": f"{d['cls']} via {d['ctx']} ({d['id']})",
                                "url": turl, "command": r.get("command")})
    return results

def run_active_checks(target: str, store, scans: list, delay: float, enabled: bool,
                      intrusive: bool = False, max_paths: int = 45, probe_budget: int = 150):
    """Coordinate the non-destructive active detection phase."""
    findings = {"jwt": [], "secrets": store.secret_findings, "takeover": [],
                "exposed_files": [], "graphql_introspection": [], "bypass_403": [],
                "passive": [], "host_checks": {}, "source_maps": [], "backup_files": [],
                "signatures": [], "header_findings": [], "fingerprints": [],
                "path_exposure": [], "active_probes": []}
    if not enabled:
        return findings

    origin = base_origin(target)
    info("Probing for exposed sensitive files/paths")
    findings["exposed_files"] = check_exposed_files(origin, delay)
    ok(f"Exposed-file leads: {len(findings['exposed_files'])}")

    # Host-level detection (security.txt, OIDC, CORS null/naive, TRACE, host-header, panels)
    info("Host-level checks (CORS/TRACE/host-header/OIDC/panels)")
    findings["host_checks"] = host_level_checks(target, store, delay)

    # JS source maps + backup/temp files (bounded)
    info("Checking JS source maps and backup/temp files")
    findings["source_maps"] = check_source_maps(store, delay)
    findings["backup_files"] = check_backup_files(store, delay)

    # Sensitive path/file exposure engine (~45 curated paths)
    info(f"Sensitive path exposure engine ({min(max_paths, len(SENSITIVE_PATHS))} paths)")
    findings["path_exposure"] = check_sensitive_paths(origin, delay, max_paths)
    ok(f"Path-exposure leads: {len(findings['path_exposure'])}")

    # GraphQL introspection on discovered graphql endpoints
    gql = [d["url"] for d in store.sorted() if "graphql" in d["tags"]][:5]
    for g in gql:
        info(f"GraphQL introspection check: {g}")
        findings["graphql_introspection"].append(check_graphql_introspection(g, delay))

    # Per-scan checks: JWT, takeover, 403 bypass, passive + signature/header/fingerprint engines
    fp_seen = set()
    for s in scans:
        base = s.get("tests", {}).get("get", {})
        jwts = find_jwt_findings(base)
        if jwts:
            findings["jwt"].append({"url": s["url"], "tokens": jwts})
            if "JWT_WEAKNESS" not in s.get("flags", []):
                s.setdefault("flags", []).append("JWT_WEAKNESS")
        tk = check_takeover(base.get("body_text", ""))
        if tk:
            findings["takeover"].append({"url": s["url"], "hits": tk})
            s.setdefault("flags", []).append("TAKEOVER_FINGERPRINT")
        code = int(base.get("meta", {}).get("http_code") or 0)
        if code in (401, 403):
            info(f"403/401 bypass surface check: {s['url']}")
            b = check_403_bypass(s["url"], code, delay)
            if b:
                findings["bypass_403"].append({"url": s["url"], "attempts": b})
                s.setdefault("flags", []).append("ACCESS_CONTROL_BYPASS_LEAD")
        pf = passive_scan_findings(s)
        if pf:
            findings["passive"].append({"url": s["url"], "items": pf})
            s["passive"] = pf
            s.setdefault("flags", []).append("PASSIVE_DETECTIONS")
        # Signature engine (SQL errors, stack traces, debug pages, info leaks)
        sig = scan_body_signatures(s)
        if sig:
            findings["signatures"].append({"url": s["url"], "items": sig})
            s.setdefault("flags", []).append("SIGNATURE_MATCH")
            if any(x["category"] == "sql-error" for x in sig):
                s.setdefault("flags", []).append("SQL_ERROR_DISCLOSURE")
        hr = scan_header_rules(s)
        if hr:
            findings["header_findings"].append({"url": s["url"], "items": hr})
        # Technology / WAF / CDN fingerprints (deduped across host)
        for fp in fingerprint_tech(s):
            key = fp["name"]
            if key not in fp_seen:
                fp_seen.add(key)
                findings["fingerprints"].append(fp)
    ok("Detected technologies: " + (", ".join(f["name"] for f in findings["fingerprints"]) or "none"))

    # Intrusive active probes (OPT-IN): 40-method active detection library
    if intrusive:
        warn(f"Intrusive probes enabled (--intrusive): {len(ACTIVE_PROBE_DEFS)} methods, "
             f"benign payloads, budget={probe_budget} requests, non-destructive")
        budget = [probe_budget]
        probe_targets = [s for s in scans if query_keys(s["url"])][:12]
        for s in probe_targets:
            if budget[0] <= 0:
                warn(f"Probe budget exhausted; {len([x for x in probe_targets])} target(s) not fully covered")
                break
            info(f"Active probes ({budget[0]} req left): {s['url']}")
            pr = active_probes(s, delay, budget=budget)
            if pr:
                findings["active_probes"].append({"url": s["url"], "probes": pr})
                s.setdefault("flags", []).append("INJECTION_PROBE_LEAD")
    return findings

# ---------------------------------------------------------------------------
# Discovery store
# ---------------------------------------------------------------------------

class DiscoveryStore:
    def __init__(self, target: str, max_urls: int):
        self.target, self.max_urls, self.urls, self.forms = target, max_urls, {}, []
        self.secret_findings = []
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
        body = resp.get("body_text", "")
        store.add_many(extract_js_endpoints(js, body), "javascript")
        for sec in scan_secrets(body, js):
            store.secret_findings.append(sec)

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
    "JWT_WEAKNESS": "[T6/P41/P42] JWT weaknesses detected — verify signature validation (alg confusion, weak secret, jku/kid) using a token you own; never send forged tokens against others.",
    "TAKEOVER_FINGERPRINT": "[subdomain-takeover] Dangling-service fingerprint in the response; confirm the CNAME target is unclaimed, then claim it only if scope allows.",
    "ACCESS_CONTROL_BYPASS_LEAD": "[T5] A path/header variant changed the 401/403 status — reproduce manually and confirm the exposed content is actually sensitive.",
    "PASSIVE_DETECTIONS": "[P81-P96] Passive signals detected (see per-endpoint list): triage each — verbose errors, tech/version disclosure, cacheable JSON, weak headers, secrets/PII in URL — and tie to concrete impact before reporting.",
    "SIGNATURE_MATCH": "[P81/T14] Response matched an error/leak signature — review the disclosed detail (stack trace, path, key) and assess exploitability.",
    "SQL_ERROR_DISCLOSURE": "[T14/P56] A database error was disclosed in the response — a strong SQLi indicator; verify manually and non-destructively with your own inputs.",
    "INJECTION_PROBE_LEAD": "[T11-T14] A benign injection probe reflected/evaluated — confirm the class manually in the right context; do not run destructive payloads against production.",
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
    af = data.get("active_findings", {})
    if af and any(af.values()):
        md += ["", "## Active detection findings (non-destructive leads)", ""]
        for ef in af.get("exposed_files", []):
            md.append(f"- **Exposed file** `{ef['url']}` (HTTP {ef['status']}, sig `{ef['signature']}`) — {ef['technique']}")
        for gi in af.get("graphql_introspection", []):
            state = "ENABLED" if gi.get("introspection_enabled") else "disabled"
            md.append(f"- **GraphQL introspection** `{gi['url']}` — **{state}** (HTTP {gi.get('status')}) — {gi['technique']}")
        for jw in af.get("jwt", []):
            for tok in jw["tokens"]:
                md.append(f"- **JWT weakness** `{jw['url']}` `{tok['token_prefix']}` — {', '.join(tok['weaknesses'])}")
        for sec in af.get("secrets", []):
            md.append(f"- **Secret** `{sec['type']}` in `{sec['source']}` (sample `{sec['sample']}`) — {sec['technique']}")
        for tk in af.get("takeover", []):
            for hit in tk["hits"]:
                md.append(f"- **Takeover fingerprint** `{tk['url']}` — {hit['provider']} (`{hit['fingerprint']}`)")
        for bp in af.get("bypass_403", []):
            for at in bp["attempts"]:
                md.append(f"- **403/401 bypass lead** `{bp['url']}` via `{at['variant']}` -> HTTP {at['status']} (baseline {at['baseline']}) — {at['technique']}")
        for sm in af.get("source_maps", []):
            md.append(f"- **Source map exposed** `{sm['url']}` — {sm['technique']}")
        for bf in af.get("backup_files", []):
            md.append(f"- **Backup/temp file** `{bf['url']}` — {bf['technique']}")
        hc = af.get("host_checks", {})
        for key, val in hc.items():
            md.append(f"- **Host check `{key}`** — `{json.dumps(val, ensure_ascii=False)[:220]}`")
        for pe in af.get("path_exposure", []):
            md.append(f"- **Exposed path** `[{pe['severity']}]` `{pe['url']}` (HTTP {pe['status']}, {pe['size']}b) — {pe['technique']}")
        if af.get("fingerprints"):
            md.append("- **Tech/WAF/CDN fingerprints**: " + ", ".join(f"{f['name']} ({f['kind']})" for f in af["fingerprints"]))
        for sg in af.get("signatures", []):
            for it in sg["items"]:
                md.append(f"- **Signature** `[{it['severity']}]` `{sg['url']}` — {it['title']} ({it['category']}, {it['technique']}) match=`{it['match']}`")
        for hf in af.get("header_findings", []):
            for it in hf["items"]:
                md.append(f"- **Header** `{hf['url']}` — {it['title']}: `{it['header']}: {it['value']}` ({it['technique']})")
        for ap_ in af.get("active_probes", []):
            for pr in ap_["probes"]:
                ctx = pr.get("context", "")
                md.append(f"- **Active probe [{pr['class']}]** `{ap_['url']}` param `{pr['parameter']}` — {ctx} ({pr.get('method','')}, {pr['technique']})")
        pv = af.get("passive", [])
        if pv:
            md += ["", "### Passive detections (per endpoint)", ""]
            for entry in pv:
                labels = ", ".join(f"{it['type']}[{it['technique']}]" for it in entry["items"])
                md.append(f"- `{entry['url']}` — {labels}")
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
        "- `RED_TEAM_METHODOLOGY.md` — 30 advanced attacker-mindset techniques (repo root)",
        "- `RED_TEAM_50_ADVANCED_PLAYS.md` — 50 deeper advanced attacker plays (repo root)", "",
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
# Interactive verification mode (--verify)
# ---------------------------------------------------------------------------

def run_verify(data: dict, delay: float):
    """Walk CRITICAL/HIGH leads and let the operator confirm each safe re-check.

    Only ever re-issues the non-destructive baseline GET, and only after an
    explicit 'y'. Prints the ready curl command + playbook steps for manual work.
    """
    if not interactive_available():
        warn("--verify needs an interactive terminal; skipping.")
        return
    scans = data.get("scans", [])
    leads = [s for s in scans if s.get("tier") in ("CRITICAL", "HIGH") or s.get("flags")]
    leads.sort(key=lambda x: (TIER_ORDER.get(x.get("tier", "INFO"), 9), -x.get("score", 0)))
    playbook_by_url = {e["url"]: e for e in data.get("playbook", [])}

    phase_banner("", "VERIFY", "Step through prioritized leads (safe GET only, on confirm)")
    if not leads:
        warn("No prioritized leads to verify.")
        return
    print(c(f"  {len(leads)} lead(s). For each: [y]=re-run safe GET, [s]=skip, [q]=quit.\n", "dim"))

    for i, s in enumerate(leads, 1):
        print(c(f"\n[{i}/{len(leads)}] [{s.get('tier')}] {s['url']}", "bold"))
        print(f"    flags   : {c(', '.join(s.get('flags', [])) or 'none', 'yellow')}")
        print(f"    http    : {s.get('summary', {}).get('http')}  ·  {s.get('summary', {}).get('content_type')}")
        pf = s.get("passive", [])
        if pf:
            print(f"    passive : {', '.join(it['type'] for it in pf)}")
        entry = playbook_by_url.get(s["url"])
        if entry:
            print(c("    manual verification:", "cyan"))
            for step in entry["steps"]:
                print(f"      - {step}")
        get_cmd = s.get("tests", {}).get("get", {}).get("command")
        if get_cmd:
            print(c("    curl:", "cyan"))
            print(f"      {get_cmd}")
        try:
            choice = input(c("    re-run safe GET now? [y/s/q]: ", "yellow")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if choice == "q":
            break
        if choice == "y":
            time.sleep(delay)
            r = curl_request(s["url"], "GET")
            print(f"      -> HTTP {r.get('meta', {}).get('http_code')} · "
                  f"{first_header(r, 'content-type')} · {r.get('meta', {}).get('size_download')} bytes")
    ok("Verification walkthrough complete.")

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
    ap.add_argument("--no-active-checks", action="store_true",
                    help="Skip the non-destructive active detection phase (JWT/secrets/exposed files/GraphQL/403-bypass)")
    ap.add_argument("--intrusive", action="store_true",
                    help="Enable benign-marker active probes (reflection/SQL-error/SSTI/LFI). Sends probe values; use only on authorized targets")
    ap.add_argument("--max-paths", type=int, default=45,
                    help="Max sensitive paths to probe in the exposure engine (default: 45)")
    ap.add_argument("--probe-budget", type=int, default=150,
                    help="Max active-probe requests when --intrusive (default: 150; prevents flooding)")
    ap.add_argument("--out", help="Output directory (skips the interactive save-location prompt)")
    ap.add_argument("--formats", help=f"Comma-separated report formats ({', '.join(VALID_FORMATS)})")
    ap.add_argument("--operator", help="Operator / handle recorded in the report")
    ap.add_argument("--engagement", help="Engagement name recorded in the report")
    ap.add_argument("--yes", action="store_true", help="Non-interactive: assume authorization and defaults")
    ap.add_argument("--verify", action="store_true",
                    help="After reporting, interactively step through CRITICAL/HIGH leads (safe GET only, on confirm)")
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

    # PHASE 5 — Active detection (non-destructive; detect, do not exploit)
    phase_banner(5, "ACTIVE DETECTION", "JWT / secrets / exposed files / GraphQL / 403-bypass")
    if args.no_active_checks:
        warn("Active detection skipped (--no-active-checks)")
        active_findings = run_active_checks(target, store, scans, args.delay, enabled=False)
    else:
        active_findings = run_active_checks(target, store, scans, args.delay, enabled=True,
                                            intrusive=args.intrusive, max_paths=args.max_paths,
                                            probe_budget=args.probe_budget)
        summary = {k: len(v) for k, v in active_findings.items()}
        ok("Active detection leads: " + ", ".join(f"{k}={n}" for k, n in summary.items() if n))

    # PHASE 6 — Analysis / playbook
    phase_banner(6, "ANALYSIS", "Prioritization and manual-test playbook")
    playbook = build_playbook(scans)
    ok(f"Playbook leads generated: {len(playbook)}")

    # PHASE 7 — Reporting
    phase_banner(7, "REPORTING", f"Writing outputs to {out}")
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
        "active_findings": active_findings,
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

    # Optional interactive verification walkthrough
    if args.verify:
        run_verify(data, args.delay)

if __name__ == "__main__":
    main()
