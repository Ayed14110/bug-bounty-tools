#!/usr/bin/env python3
"""
ayed_recon.py  -  Authorized Reconnaissance, OS Detection & CVE Correlation Tool
================================================================================

Given a target host/URL this tool performs:

  1. Information gathering   -> DNS, reverse-DNS, WHOIS, HTTP fingerprint,
                               TLS/SSL certificate, security-header audit,
                               technology detection, common-port scan.
  2. OS detection            -> nmap -O when available (root), otherwise a
                               passive heuristic based on TTL + Server header.
  3. CVE correlation         -> maps detected products/versions to:
                                 * NVD / CVE (live keyword query, 2026-aware)
                                 * OWASP Top 10 (2021 categories + CWE map)
                                 * Exploit-DB search links
                                 * Google Hacking DB (GHDB) dork suggestions
  4. Professional report     -> Markdown + self-contained HTML report saved
                               under ./reports/.

Designed for Kali Linux. Uses nmap / whois when present and falls back to
pure-Python methods so it still runs in a minimal environment.

USAGE
    python3 ayed_recon.py testaspnet.vulnweb.com
    python3 ayed_recon.py https://example.com --ports 1-1000 --no-cve

LEGAL / ETHICAL NOTICE
    Only run this against systems you own or are explicitly authorized to
    test. testaspnet.vulnweb.com is a public, intentionally-vulnerable test
    site provided by Acunetix for exactly this purpose.

Author: ayedcyper
"""

import argparse
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from html import escape
from urllib.parse import urlparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import urllib3
    urllib3.disable_warnings()
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


# --------------------------------------------------------------------------- #
#  Pretty terminal output (no external deps)
# --------------------------------------------------------------------------- #
class C:
    R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"
    M = "\033[95m"; CY = "\033[96m"; W = "\033[97m"; D = "\033[2m"
    BOLD = "\033[1m"; END = "\033[0m"


def banner():
    art = r"""
   ___                 ______
  / _ \  __ _ _   _  __| /  __\___  _ __  ___ _ __
 / /_)/ / _` | | | |/ _` | / __/ _ \| '_ \/ __| '__|
/ ___/ | (_| | |_| | (_| || (_|  __/| |_) \__ \ |
\/      \__,_|\__, |\__,_| \___\___|| .__/|___/_|
              |___/                 |_|
        Recon + OS Detect + CVE Correlation  ::  ayedcyper
"""
    print(C.CY + art + C.END)


def info(m):  print(f"{C.B}[*]{C.END} {m}")
def good(m):  print(f"{C.G}[+]{C.END} {m}")
def warn(m):  print(f"{C.Y}[!]{C.END} {m}")
def bad(m):   print(f"{C.R}[-]{C.END} {m}")
def sect(m):  print(f"\n{C.M}{C.BOLD}==== {m} ===={C.END}")


# --------------------------------------------------------------------------- #
#  HTTP session
# --------------------------------------------------------------------------- #
def build_session():
    if not REQUESTS_OK:
        return None
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.4,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
                       "ayed_recon/1.0")
    })
    return s


# --------------------------------------------------------------------------- #
#  Target normalisation
# --------------------------------------------------------------------------- #
def normalize(target):
    if "://" not in target:
        target = "http://" + target
    p = urlparse(target)
    host = p.hostname
    scheme = p.scheme
    port = p.port or (443 if scheme == "https" else 80)
    return host, scheme, port, target


# --------------------------------------------------------------------------- #
#  1. DNS / reverse DNS
# --------------------------------------------------------------------------- #
def dns_info(host):
    out = {"a_records": [], "reverse": {}, "cname": None}
    try:
        infos = socket.getaddrinfo(host, None)
        ips = sorted({i[4][0] for i in infos})
        out["a_records"] = ips
        for ip in ips:
            try:
                out["reverse"][ip] = socket.gethostbyaddr(ip)[0]
            except Exception:
                out["reverse"][ip] = None
    except Exception as e:
        out["error"] = str(e)
    return out


# --------------------------------------------------------------------------- #
#  2. WHOIS (system binary if present, else RDAP over HTTPS)
# --------------------------------------------------------------------------- #
def whois_info(host, session):
    if shutil.which("whois"):
        try:
            r = subprocess.run(["whois", host], capture_output=True,
                               text=True, timeout=25)
            return {"source": "whois", "raw": r.stdout[:6000]}
        except Exception as e:
            return {"source": "whois", "error": str(e)}
    # RDAP fallback (no binary needed)
    if session:
        base = host.split(".")[-2] + "." + host.split(".")[-1] if host.count(".") >= 1 else host
        try:
            r = session.get(f"https://rdap.org/domain/{base}", timeout=15, verify=False)
            if r.ok:
                d = r.json()
                events = {e.get("eventAction"): e.get("eventDate")
                          for e in d.get("events", [])}
                return {"source": "rdap",
                        "domain": d.get("ldhName"),
                        "status": d.get("status"),
                        "registered": events.get("registration"),
                        "expires": events.get("expiration"),
                        "nameservers": [n.get("ldhName")
                                        for n in d.get("nameservers", [])]}
        except Exception as e:
            return {"source": "rdap", "error": str(e)}
    return {"error": "no whois source available"}


# --------------------------------------------------------------------------- #
#  3. HTTP fingerprint + security headers + tech detection
# --------------------------------------------------------------------------- #
SEC_HEADERS = [
    "Strict-Transport-Security", "Content-Security-Policy", "X-Frame-Options",
    "X-Content-Type-Options", "Referrer-Policy", "Permissions-Policy",
    "X-XSS-Protection",
]

TECH_SIGNATURES = {
    "Microsoft-IIS":     ("Server", r"Microsoft-IIS/([\d.]+)", "Microsoft IIS"),
    "ASP.NET":           ("X-Powered-By", r"ASP\.NET", "Microsoft ASP.NET"),
    "AspNetVersion":     ("X-AspNet-Version", r"([\d.]+)", "ASP.NET Framework"),
    "AspNetMvc":         ("X-AspNetMvc-Version", r"([\d.]+)", "ASP.NET MVC"),
    "Apache":            ("Server", r"Apache/?([\d.]+)?", "Apache httpd"),
    "nginx":             ("Server", r"nginx/?([\d.]+)?", "nginx"),
    "PHP":               ("X-Powered-By", r"PHP/?([\d.]+)?", "PHP"),
    "Express":           ("X-Powered-By", r"Express", "Node.js Express"),
}


def http_fingerprint(url, session):
    out = {"url": url, "reachable": False}
    if not session:
        out["error"] = "requests library unavailable"
        return out
    try:
        r = session.get(url, timeout=15, verify=False, allow_redirects=True)
    except Exception as e:
        out["error"] = str(e)
        return out

    out["reachable"] = True
    out["status_code"] = r.status_code
    out["final_url"] = r.url
    out["headers"] = dict(r.headers)
    out["redirect_chain"] = [h.url for h in r.history] + [r.url]

    # title
    m = re.search(r"<title[^>]*>(.*?)</title>", r.text or "",
                  re.I | re.S)
    out["title"] = m.group(1).strip()[:200] if m else None

    # cookies + flags
    cookies = []
    for ck in r.cookies:
        cookies.append({"name": ck.name,
                        "secure": ck.secure,
                        "httponly": bool(ck._rest.get("HttpOnly", False)
                                         if hasattr(ck, "_rest") else False)})
    out["cookies"] = cookies

    # security-header audit
    present, missing = {}, []
    for h in SEC_HEADERS:
        if h in r.headers:
            present[h] = r.headers[h]
        else:
            missing.append(h)
    out["security_headers"] = {"present": present, "missing": missing}

    # tech detection
    tech = []
    body = r.text or ""
    for _, (hdr, pat, label) in TECH_SIGNATURES.items():
        val = r.headers.get(hdr, "")
        m = re.search(pat, val, re.I)
        if m:
            ver = m.group(1) if m.groups() and m.group(1) else None
            tech.append({"product": label,
                         "version": ver,
                         "evidence": f"{hdr}: {val}".strip()})
    mg = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)',
                   body, re.I)
    if mg:
        tech.append({"product": "Generator: " + mg.group(1),
                     "version": None, "evidence": "meta generator tag"})
    # dedupe
    seen, uniq = set(), []
    for t in tech:
        k = (t["product"], t["version"])
        if k not in seen:
            seen.add(k); uniq.append(t)
    out["technologies"] = uniq
    return out


# --------------------------------------------------------------------------- #
#  4. TLS / certificate
# --------------------------------------------------------------------------- #
def tls_info(host, port=443):
    out = {}
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ss:
                cert = ss.getpeercert()
                out["protocol"] = ss.version()
                out["cipher"] = ss.cipher()[0] if ss.cipher() else None
                if cert:
                    out["subject"] = dict(x[0] for x in cert.get("subject", []))
                    out["issuer"] = dict(x[0] for x in cert.get("issuer", []))
                    out["not_after"] = cert.get("notAfter")
                    out["not_before"] = cert.get("notBefore")
                    # expiry check
                    try:
                        exp = datetime.strptime(cert["notAfter"],
                                                "%b %d %H:%M:%S %Y %Z")
                        out["expired"] = exp < datetime.utcnow()
                        out["days_left"] = (exp - datetime.utcnow()).days
                    except Exception:
                        pass
    except Exception as e:
        out["error"] = str(e)
    return out


# --------------------------------------------------------------------------- #
#  5. Port scan  (nmap if available, else fast socket scan)
# --------------------------------------------------------------------------- #
COMMON_PORTS = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    80: "http", 110: "pop3", 135: "msrpc", 139: "netbios",
    143: "imap", 443: "https", 445: "smb", 993: "imaps",
    995: "pop3s", 1433: "mssql", 3306: "mysql", 3389: "rdp",
    5432: "postgres", 5985: "winrm", 6379: "redis", 8080: "http-alt",
    8443: "https-alt", 27017: "mongodb",
}


def socket_scan(ip, ports):
    open_ports = []
    for p in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.2)
        try:
            if s.connect_ex((ip, p)) == 0:
                svc = COMMON_PORTS.get(p, "unknown")
                # light banner grab
                banner_txt = ""
                try:
                    s.settimeout(1.5)
                    if p in (80, 8080, 8443, 443):
                        s.sendall(b"HEAD / HTTP/1.0\r\n\r\n")
                    banner_txt = s.recv(120).decode("latin-1", "ignore").strip()
                except Exception:
                    pass
                open_ports.append({"port": p, "service": svc,
                                   "banner": banner_txt[:120]})
        except Exception:
            pass
        finally:
            s.close()
    return open_ports


def port_scan(host, ip, port_spec):
    # nmap path
    if shutil.which("nmap"):
        info("nmap detected — running service/version scan")
        try:
            cmd = ["nmap", "-sV", "-Pn", "--host-timeout", "120s",
                   "-p", port_spec, ip]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            return {"scanner": "nmap", "raw": r.stdout,
                    "open": _parse_nmap_ports(r.stdout)}
        except Exception as e:
            warn(f"nmap failed ({e}); falling back to socket scan")
    # socket fallback
    if "-" in port_spec:
        a, b = port_spec.split("-")
        ports = list(range(int(a), min(int(b), int(a) + 1024) + 1))
    else:
        ports = [int(x) for x in port_spec.split(",")] if port_spec else list(COMMON_PORTS)
    ports = ports or list(COMMON_PORTS)
    return {"scanner": "socket", "open": socket_scan(ip, ports)}


def _parse_nmap_ports(text):
    res = []
    for line in text.splitlines():
        m = re.match(r"(\d+)/tcp\s+open\s+(\S+)\s*(.*)", line)
        if m:
            res.append({"port": int(m.group(1)), "service": m.group(2),
                        "banner": m.group(3).strip()})
    return res


# --------------------------------------------------------------------------- #
#  6. OS detection
# --------------------------------------------------------------------------- #
def os_detection(host, ip, fingerprint):
    result = {"method": None, "guess": None, "confidence": "low", "evidence": []}

    # nmap -O needs root
    if shutil.which("nmap") and os.geteuid() == 0:
        try:
            r = subprocess.run(["nmap", "-O", "-Pn", "--host-timeout", "90s", ip],
                              capture_output=True, text=True, timeout=140)
            m = re.search(r"OS details: (.+)", r.stdout)
            g = re.search(r"Running: (.+)", r.stdout)
            if m or g:
                result.update(method="nmap -O",
                              guess=(m.group(1) if m else g.group(1)),
                              confidence="high",
                              evidence=["nmap TCP/IP stack fingerprint"])
                return result
        except Exception:
            pass

    # Passive heuristic: Server header + TTL
    server = (fingerprint.get("headers", {}) or {}).get("Server", "")
    powered = (fingerprint.get("headers", {}) or {}).get("X-Powered-By", "")
    ev, guess = [], None
    if re.search(r"IIS|ASP\.NET", server + powered, re.I):
        guess = "Windows Server (IIS / ASP.NET stack)"; ev.append(f"Server/Powered-By: {server} {powered}".strip())
    elif re.search(r"Ubuntu|Debian", server, re.I):
        guess = "Linux (Debian/Ubuntu)"; ev.append(f"Server: {server}")
    elif re.search(r"CentOS|Red Hat|Fedora", server, re.I):
        guess = "Linux (RHEL/CentOS)"; ev.append(f"Server: {server}")
    elif re.search(r"nginx|Apache", server, re.I):
        guess = "Unix/Linux (nginx/Apache — distro not disclosed)"; ev.append(f"Server: {server}")

    # TTL via ping
    ttl = _ping_ttl(ip)
    if ttl:
        ev.append(f"ICMP TTL={ttl}")
        if ttl <= 64 and not guess:
            guess = "Linux/Unix (TTL≈64)"
        elif ttl > 64 and ttl <= 128:
            if not guess:
                guess = "Windows (TTL≈128)"
    result.update(method="passive (headers/TTL)", guess=guess or "Undetermined",
                  confidence="medium" if guess else "low", evidence=ev)
    return result


def _ping_ttl(ip):
    if not shutil.which("ping"):
        return None
    try:
        r = subprocess.run(["ping", "-c", "1", "-W", "2", ip],
                          capture_output=True, text=True, timeout=6)
        m = re.search(r"ttl=(\d+)", r.stdout, re.I)
        return int(m.group(1)) if m else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  7. CVE / OWASP / ExploitDB / GHDB correlation
# --------------------------------------------------------------------------- #
# OWASP Top 10 (2021, still current baseline) + CWE mapping.
OWASP_TOP10 = [
    ("A01:2021", "Broken Access Control", ["CWE-200", "CWE-284", "CWE-639"]),
    ("A02:2021", "Cryptographic Failures", ["CWE-259", "CWE-327", "CWE-319"]),
    ("A03:2021", "Injection (SQLi/XSS/Cmd)", ["CWE-79", "CWE-89", "CWE-77"]),
    ("A04:2021", "Insecure Design", ["CWE-209", "CWE-256", "CWE-501"]),
    ("A05:2021", "Security Misconfiguration", ["CWE-16", "CWE-611", "CWE-548"]),
    ("A06:2021", "Vulnerable & Outdated Components", ["CWE-1104", "CWE-937"]),
    ("A07:2021", "Identification & Auth Failures", ["CWE-287", "CWE-384", "CWE-620"]),
    ("A08:2021", "Software & Data Integrity Failures", ["CWE-502", "CWE-829"]),
    ("A09:2021", "Security Logging & Monitoring Failures", ["CWE-778", "CWE-117"]),
    ("A10:2021", "Server-Side Request Forgery (SSRF)", ["CWE-918"]),
]


def owasp_map_from_findings(fingerprint, tls):
    """Flag OWASP categories evidenced by passive findings."""
    flags = []
    sh = fingerprint.get("security_headers", {})
    missing = sh.get("missing", []) if sh else []
    if missing:
        flags.append(("A05:2021", "Security Misconfiguration",
                      f"Missing security headers: {', '.join(missing)}"))
    # cookie flags
    for ck in fingerprint.get("cookies", []):
        if not ck.get("secure") or not ck.get("httponly"):
            flags.append(("A05:2021", "Security Misconfiguration",
                          f"Cookie '{ck['name']}' missing Secure/HttpOnly flag"))
            break
    # crypto
    if tls.get("expired"):
        flags.append(("A02:2021", "Cryptographic Failures", "TLS certificate expired"))
    proto = tls.get("protocol", "")
    if proto in ("TLSv1", "TLSv1.1", "SSLv3"):
        flags.append(("A02:2021", "Cryptographic Failures",
                      f"Weak TLS protocol negotiated: {proto}"))
    if fingerprint.get("final_url", "").startswith("http://"):
        flags.append(("A02:2021", "Cryptographic Failures",
                      "Site served over cleartext HTTP"))
    # outdated components (any versioned tech)
    for t in fingerprint.get("technologies", []):
        if t.get("version"):
            flags.append(("A06:2021", "Vulnerable & Outdated Components",
                          f"{t['product']} version {t['version']} disclosed"))
    return flags


def nvd_lookup(session, keyword, limit=5):
    """Live NVD 2.0 keyword search, newest first. Best-effort."""
    if not session:
        return {"error": "no session"}
    try:
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {"keywordSearch": keyword, "resultsPerPage": limit}
        r = session.get(url, params=params, timeout=25, verify=False)
        if not r.ok:
            return {"error": f"HTTP {r.status_code}"}
        data = r.json()
        cves = []
        for item in data.get("vulnerabilities", [])[:limit]:
            c = item.get("cve", {})
            desc = ""
            for d in c.get("descriptions", []):
                if d.get("lang") == "en":
                    desc = d.get("value", ""); break
            metrics = c.get("metrics", {})
            score = None
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if key in metrics and metrics[key]:
                    score = metrics[key][0]["cvssData"].get("baseScore")
                    break
            cves.append({"id": c.get("id"),
                         "published": c.get("published", "")[:10],
                         "score": score,
                         "summary": desc[:240]})
        return {"keyword": keyword, "results": cves}
    except Exception as e:
        return {"error": str(e)}


def cve_correlation(fingerprint, session, do_live=True):
    products = []
    for t in fingerprint.get("technologies", []):
        name = t["product"]
        kw = name
        if t.get("version"):
            kw = f"{name} {t['version']}"
        products.append({"product": name, "version": t.get("version"),
                         "keyword": kw})

    correlation = []
    for p in products:
        entry = {"product": p["product"], "version": p["version"]}
        # search links (always available, offline)
        q = requests.utils.quote(p["keyword"]) if REQUESTS_OK else p["keyword"].replace(" ", "+")
        entry["links"] = {
            "nvd": f"https://nvd.nist.gov/vuln/search/results?query={q}",
            "exploitdb": f"https://www.exploit-db.com/search?q={q}",
            "ghdb": "https://www.exploit-db.com/google-hacking-database",
            "cve_mitre": f"https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword={q}",
        }
        # live NVD (2026-aware, newest first)
        if do_live and session:
            entry["nvd_live"] = nvd_lookup(session, p["keyword"])
            time.sleep(0.7)  # be polite to NVD rate limit
        correlation.append(entry)
    return {"products": products, "correlation": correlation}


def ghdb_dorks(host):
    """Google Hacking DB style dorks tailored to the target."""
    return [
        f'site:{host} inurl:admin',
        f'site:{host} inurl:login',
        f'site:{host} filetype:aspx',
        f'site:{host} inurl:"?id="',
        f'site:{host} intitle:"index of"',
        f'site:{host} ext:config OR ext:bak OR ext:old',
        f'site:{host} "sql syntax near" OR "Warning: mysql"',
        f'site:{host} inurl:web.config',
    ]


# --------------------------------------------------------------------------- #
#  Report generation
# --------------------------------------------------------------------------- #
def generate_report(target, data, outdir):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", data["host"])
    md_path = os.path.join(outdir, f"recon_{safe}_{ts}.md")
    html_path = os.path.join(outdir, f"recon_{safe}_{ts}.html")

    md = _build_markdown(target, data, ts)
    with open(md_path, "w") as f:
        f.write(md)
    with open(html_path, "w") as f:
        f.write(_build_html(target, data, md, ts))
    return md_path, html_path


def _build_markdown(target, d, ts):
    L = []
    A = L.append
    A(f"# Reconnaissance & Vulnerability Correlation Report")
    A(f"\n**Target:** `{target}`  ")
    A(f"**Host:** `{d['host']}`  ")
    A(f"**Assessment date (UTC):** {ts}  ")
    A(f"**Assessor:** ayedcyper  ")
    A(f"**Authorization:** Public test target (Acunetix vulnweb) — authorized.\n")

    A("## 1. Executive Summary\n")
    fp = d["http"]
    techs = ", ".join(t["product"] + (f" {t['version']}" if t['version'] else "")
                      for t in fp.get("technologies", [])) or "none disclosed"
    A(f"- **Reachable:** {fp.get('reachable')} (HTTP {fp.get('status_code','-')})")
    A(f"- **Detected stack:** {techs}")
    A(f"- **OS guess:** {d['os']['guess']} ({d['os']['confidence']} confidence)")
    missing = fp.get("security_headers", {}).get("missing", [])
    A(f"- **Missing security headers:** {len(missing)}")
    open_ports = d["ports"].get("open", [])
    A(f"- **Open ports found:** {len(open_ports)}")
    owasp = d["owasp_flags"]
    A(f"- **OWASP categories flagged (passive):** {len(owasp)}\n")

    A("## 2. Information Gathering\n")
    A("### 2.1 DNS")
    dns = d["dns"]
    for ip in dns.get("a_records", []):
        rev = dns.get("reverse", {}).get(ip)
        A(f"- `{ip}`" + (f"  ->  PTR `{rev}`" if rev else ""))
    if not dns.get("a_records"):
        A("- No A records resolved.")

    A("\n### 2.2 WHOIS / RDAP")
    w = d["whois"]
    if w.get("source") == "rdap":
        A(f"- Domain: `{w.get('domain')}`")
        A(f"- Registered: {w.get('registered')}  |  Expires: {w.get('expires')}")
        A(f"- Nameservers: {', '.join(w.get('nameservers') or []) or 'n/a'}")
        A(f"- Status: {', '.join(w.get('status') or []) or 'n/a'}")
    elif w.get("raw"):
        A("```\n" + w["raw"][:1500] + "\n```")
    else:
        A(f"- {w.get('error','n/a')}")

    A("\n### 2.3 HTTP Fingerprint")
    A(f"- Final URL: `{fp.get('final_url','-')}`")
    A(f"- Title: {fp.get('title')}")
    if fp.get("redirect_chain") and len(fp["redirect_chain"]) > 1:
        A(f"- Redirect chain: {' -> '.join(fp['redirect_chain'])}")
    A("\n**Response headers:**\n")
    A("```")
    for k, v in (fp.get("headers", {}) or {}).items():
        A(f"{k}: {v}")
    A("```")

    A("\n### 2.4 Security Header Audit")
    sh = fp.get("security_headers", {})
    for h, v in (sh.get("present") or {}).items():
        A(f"- ✅ `{h}`: {v}")
    for h in (sh.get("missing") or []):
        A(f"- ❌ Missing `{h}`")

    A("\n### 2.5 Cookies")
    for ck in fp.get("cookies", []):
        A(f"- `{ck['name']}` Secure={ck['secure']} HttpOnly={ck['httponly']}")
    if not fp.get("cookies"):
        A("- No cookies set.")

    A("\n### 2.6 Technology Detection")
    for t in fp.get("technologies", []):
        A(f"- **{t['product']}** {t['version'] or ''} — _{t['evidence']}_")
    if not fp.get("technologies"):
        A("- No technologies fingerprinted from headers/body.")

    A("\n### 2.7 TLS / Certificate")
    tls = d["tls"]
    if tls.get("error"):
        A(f"- {tls['error']}")
    else:
        A(f"- Protocol: {tls.get('protocol')}  |  Cipher: {tls.get('cipher')}")
        A(f"- Issuer: {tls.get('issuer')}")
        A(f"- Valid: {tls.get('not_before')} → {tls.get('not_after')} "
          f"(days left: {tls.get('days_left','?')}, expired: {tls.get('expired')})")

    A("\n## 3. OS Detection\n")
    osd = d["os"]
    A(f"- **Method:** {osd['method']}")
    A(f"- **Guess:** {osd['guess']}")
    A(f"- **Confidence:** {osd['confidence']}")
    for e in osd.get("evidence", []):
        A(f"- Evidence: {e}")

    A("\n## 4. Port / Service Scan\n")
    A(f"- Scanner: {d['ports'].get('scanner')}")
    if open_ports:
        A("\n| Port | Service | Banner |")
        A("|------|---------|--------|")
        for p in open_ports:
            A(f"| {p['port']} | {p.get('service','')} | "
              f"{escape(str(p.get('banner','')))[:60]} |")
    else:
        A("- No open ports detected in scanned range (or scan blocked by network policy).")

    A("\n## 5. CVE Correlation (NVD / Exploit-DB / MITRE)\n")
    cve = d["cve"]
    if not cve["correlation"]:
        A("- No versioned products available to correlate.")
    for c in cve["correlation"]:
        A(f"\n### {c['product']} {c['version'] or ''}")
        links = c["links"]
        A(f"- 🔎 NVD search: {links['nvd']}")
        A(f"- 💥 Exploit-DB: {links['exploitdb']}")
        A(f"- 📚 CVE/MITRE: {links['cve_mitre']}")
        live = c.get("nvd_live", {})
        if live.get("results"):
            A(f"\n  **Recent CVEs (NVD, newest first):**")
            A("\n  | CVE | Published | CVSS | Summary |")
            A("  |-----|-----------|------|---------|")
            for r in live["results"]:
                A(f"  | {r['id']} | {r['published']} | {r['score']} | "
                  f"{escape(str(r['summary']))[:80]} |")
        elif live.get("error"):
            A(f"  - _NVD live query: {live['error']} (use the search link above)_")

    A("\n## 6. OWASP Top 10 Mapping\n")
    A("### 6.1 Passive findings mapped to OWASP\n")
    if owasp:
        A("| OWASP | Category | Evidence |")
        A("|-------|----------|----------|")
        for cat, name, ev in owasp:
            A(f"| {cat} | {name} | {escape(ev)} |")
    else:
        A("- No OWASP categories evidenced passively.")
    A("\n### 6.2 Full OWASP Top 10 checklist (manual verification recommended)\n")
    A("| ID | Category | Related CWE |")
    A("|----|----------|-------------|")
    for cid, name, cwes in OWASP_TOP10:
        A(f"| {cid} | {name} | {', '.join(cwes)} |")

    A("\n## 7. Google Hacking (GHDB) Dorks\n")
    A("Recommended dorks for deeper OSINT on this target:\n")
    A("```")
    for dork in d["ghdb"]:
        A(dork)
    A("```")

    A("\n## 8. Recommendations\n")
    recs = []
    if missing:
        recs.append("Add the missing HTTP security headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options).")
    if any(not ck.get("secure") or not ck.get("httponly") for ck in fp.get("cookies", [])):
        recs.append("Set `Secure` and `HttpOnly` (and `SameSite`) flags on all session cookies.")
    if any(t.get("version") for t in fp.get("technologies", [])):
        recs.append("Suppress version banners (Server, X-AspNet-Version, X-Powered-By) and patch disclosed components.")
    if fp.get("final_url", "").startswith("http://"):
        recs.append("Enforce HTTPS site-wide and redirect HTTP→HTTPS.")
    recs.append("Manually validate injection points (SQLi/XSS) — this target is known-vulnerable by design.")
    recs.append("Cross-check each disclosed component version against the linked NVD/Exploit-DB results.")
    for i, r in enumerate(recs, 1):
        A(f"{i}. {r}")

    A("\n---\n_Generated by ayed_recon.py — authorized testing only._")
    return "\n".join(L)


def _build_html(target, d, md_text, ts):
    # lightweight markdown->html (headings, tables, code, lists, links)
    lines = md_text.splitlines()
    html, in_code, in_table = [], False, False
    def close_table():
        nonlocal in_table
        if in_table:
            html.append("</table>"); in_table = False
    for ln in lines:
        if ln.startswith("```"):
            close_table()
            if not in_code:
                html.append("<pre><code>"); in_code = True
            else:
                html.append("</code></pre>"); in_code = False
            continue
        if in_code:
            html.append(escape(ln)); continue
        # tables
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue  # separator row
            if not in_table:
                html.append("<table>"); in_table = True
                html.append("<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells) + "</tr>")
            else:
                html.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
            continue
        close_table()
        if ln.startswith("### "):
            html.append(f"<h3>{_inline(ln[4:])}</h3>")
        elif ln.startswith("## "):
            html.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            html.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("- "):
            html.append(f"<li>{_inline(ln[2:])}</li>")
        elif re.match(r"\d+\. ", ln):
            html.append(f"<li>{_inline(ln[ln.index('.')+2:])}</li>")
        elif ln.strip() == "---":
            html.append("<hr>")
        elif ln.strip() == "":
            html.append("<br>")
        else:
            html.append(f"<p>{_inline(ln)}</p>")
    close_table()
    body = "\n".join(html)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recon Report — {escape(d['host'])}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px;
         margin: 0 auto; padding: 2rem 1.2rem; line-height: 1.55;
         background:#0d1117; color:#e6edf3; }}
  h1 {{ border-bottom:3px solid #58a6ff; padding-bottom:.3em; color:#58a6ff; }}
  h2 {{ border-bottom:1px solid #30363d; padding-bottom:.25em; margin-top:2em; color:#79c0ff;}}
  h3 {{ color:#a5d6ff; margin-top:1.4em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:.9em; overflow-x:auto; display:block;}}
  th,td {{ border:1px solid #30363d; padding:.45em .6em; text-align:left; }}
  th {{ background:#161b22; }}
  tr:nth-child(even) td {{ background:#161b22; }}
  pre {{ background:#161b22; padding:1em; border-radius:8px; overflow-x:auto; border:1px solid #30363d;}}
  code {{ font-family: ui-monospace, Menlo, monospace; font-size:.85em; }}
  a {{ color:#58a6ff; word-break:break-all; }}
  li {{ margin:.2em 0; }}
  hr {{ border:none; border-top:1px solid #30363d; margin:2em 0; }}
</style></head><body>
{body}
</body></html>"""


def _inline(text):
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"_([^_]+)_", r"<em>\1</em>", text)
    text = re.sub(r"(https?://[^\s<]+)", r'<a href="\1">\1</a>', text)
    return text


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run(target, args):
    host, scheme, port, url = normalize(target)
    session = build_session()

    sect("TARGET")
    good(f"Target: {target}  ->  host={host} scheme={scheme} port={port}")

    sect("1. DNS / REVERSE DNS")
    dns = dns_info(host)
    ip = dns["a_records"][0] if dns.get("a_records") else host
    for a in dns.get("a_records", []):
        good(f"A {a}  PTR {dns['reverse'].get(a)}")

    sect("2. WHOIS / RDAP")
    who = whois_info(host, session)
    info(f"whois source: {who.get('source', who.get('error'))}")

    sect("3. HTTP FINGERPRINT")
    fp = http_fingerprint(url, session)
    if fp.get("reachable"):
        good(f"HTTP {fp['status_code']}  title={fp.get('title')}")
        for t in fp.get("technologies", []):
            good(f"tech: {t['product']} {t['version'] or ''}")
        for h in fp.get("security_headers", {}).get("missing", []):
            warn(f"missing header: {h}")
    else:
        bad(f"not reachable: {fp.get('error')}")

    sect("4. TLS / CERTIFICATE")
    tls = tls_info(host, 443) if scheme == "https" or port == 443 else {}
    if not tls or tls.get("error"):
        # try https anyway
        tls = tls_info(host, 443)
    info(f"TLS: {tls.get('protocol', tls.get('error','n/a'))}")

    sect("5. PORT SCAN")
    ports = port_scan(host, ip, args.ports)
    for p in ports.get("open", []):
        good(f"open {p['port']}/{p.get('service')}  {p.get('banner','')[:50]}")
    if not ports.get("open"):
        warn("no open ports (network policy may block outbound scanning)")

    sect("6. OS DETECTION")
    osd = os_detection(host, ip, fp)
    good(f"OS: {osd['guess']} ({osd['confidence']}) via {osd['method']}")

    sect("7. CVE CORRELATION")
    cve = cve_correlation(fp, session, do_live=not args.no_cve)
    for c in cve["correlation"]:
        info(f"{c['product']} {c['version'] or ''}")
        live = c.get("nvd_live", {})
        for r in live.get("results", [])[:3]:
            print(f"    {C.R}{r['id']}{C.END} CVSS={r['score']} {r['summary'][:60]}")

    sect("8. OWASP MAPPING")
    owasp = owasp_map_from_findings(fp, tls)
    for cat, name, ev in owasp:
        warn(f"{cat} {name}: {ev}")

    data = {
        "host": host, "target": target, "dns": dns, "whois": who,
        "http": fp, "tls": tls, "ports": ports, "os": osd,
        "cve": cve, "owasp_flags": owasp, "ghdb": ghdb_dorks(host),
    }

    sect("REPORT")
    outdir = args.outdir
    md_path, html_path = generate_report(target, data, outdir)
    good(f"Markdown report: {md_path}")
    good(f"HTML report:     {html_path}")
    # also dump raw json
    json_path = md_path.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    good(f"Raw JSON:        {json_path}")
    return md_path, html_path, json_path


def main():
    ap = argparse.ArgumentParser(
        description="Authorized recon + OS detect + CVE correlation + reporting.")
    ap.add_argument("target", help="target host or URL (e.g. testaspnet.vulnweb.com)")
    ap.add_argument("--ports", default="21,22,23,25,53,80,110,135,139,143,443,445,"
                    "993,995,1433,3306,3389,5432,8080,8443",
                    help="ports/range (e.g. 1-1000 or 80,443)")
    ap.add_argument("--no-cve", action="store_true", help="skip live NVD queries")
    ap.add_argument("--outdir", default="reports", help="report output directory")
    args = ap.parse_args()

    banner()
    if not REQUESTS_OK:
        bad("The 'requests' library is required: pip install requests")
        sys.exit(1)
    try:
        run(args.target, args)
    except KeyboardInterrupt:
        bad("interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
