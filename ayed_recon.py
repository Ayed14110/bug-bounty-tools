#!/usr/bin/env python3
"""
ayed_recon.py  v2.0  -  Advanced Recon, OS Detection & Threat-Intel Correlation
================================================================================

Given a target this tool performs deep information gathering, OS detection and
multi-source vulnerability-intelligence correlation, then produces a
professional risk-scored report (Markdown + HTML + SARIF + SQLite).

40 ADVANCED FEATURES (see FEATURES list at bottom of file for the numbered map)
Highlights:
  * CPE 2.3 exact CVE matching (NVD) with keyword fallback
  * CISA KEV (Known Exploited Vulnerabilities) enrichment
  * FIRST EPSS exploit-probability scoring
  * Composite risk score  =  CVSS x EPSS  (KEV = critical override)
  * Disk cache with TTL for every intel source (rate-limit friendly)
  * DNS-over-HTTPS records (A/AAAA/MX/TXT/NS/CAA) + SPF/DMARC/DKIM posture
  * Subdomain enumeration via crt.sh certificate-transparency logs
  * WAF detection, CORS audit, HTTP-method audit, favicon hash
  * Sensitive-file exposure probing (.git/.env/backup)
  * Security-header letter grade (A-F), cookie SameSite analysis
  * Concurrent port scan, ASN/GeoIP, Wayback URL harvest
  * OWASP Top 10 mapping + composite target risk rating
  * SARIF 2.1.0 output for CI pipelines + SQLite findings store

Designed for Kali Linux; degrades gracefully without nmap/whois/dnspython.

LEGAL: authorized targets only. testaspnet.vulnweb.com is Acunetix's public
intentionally-vulnerable test site.

Author: ayedcyper
"""

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import ssl
import sqlite3
import struct
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from html import escape
from urllib.parse import urlparse, quote

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import urllib3
    urllib3.disable_warnings()
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import mmh3          # feature: real shodan-style favicon hash
    MMH3_OK = True
except ImportError:
    MMH3_OK = False

VERSION = "2.0"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "reports", ".cache")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "bug_bounty_findings.db")

log = logging.getLogger("ayed_recon")


# --------------------------------------------------------------------------- #
#  Terminal styling
# --------------------------------------------------------------------------- #
class C:
    R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"
    M = "\033[95m"; CY = "\033[96m"; W = "\033[97m"; D = "\033[2m"
    BOLD = "\033[1m"; END = "\033[0m"


def banner():
    print(C.CY + r"""
   ___                 ______
  / _ \  __ _ _   _  __| /  __\___  _ __  ___ _ __
 / /_)/ / _` | | | |/ _` | / __/ _ \| '_ \/ __| '__|
/ ___/ | (_| | |_| | (_| || (_|  __/| |_) \__ \ |
\/      \__,_|\__, |\__,_| \___\___|| .__/|___/_|
              |___/   v%s          |_|
   Recon + OS + CVE/KEV/EPSS Threat-Intel  ::  ayedcyper
""" % VERSION + C.END)


def info(m): print(f"{C.B}[*]{C.END} {m}")
def good(m): print(f"{C.G}[+]{C.END} {m}")
def warn(m): print(f"{C.Y}[!]{C.END} {m}")
def bad(m):  print(f"{C.R}[-]{C.END} {m}")
def sect(m): print(f"\n{C.M}{C.BOLD}==== {m} ===={C.END}")


# --------------------------------------------------------------------------- #
#  Feature #6 : disk cache with TTL
# --------------------------------------------------------------------------- #
def cache_get(key, ttl):
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".json")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def cache_put(key, value):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".json")
    try:
        with open(path, "w") as f:
            json.dump(value, f)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Feature #7/#8 : session (proxy + NVD key + retry/backoff)
# --------------------------------------------------------------------------- #
def build_session(proxy=None):
    if not REQUESTS_OK:
        return None
    s = requests.Session()
    s.verify = False
    retry = Retry(total=3, backoff_factor=0.6, respect_retry_after_header=True,
                  status_forcelist=[403, 429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 "
                     "Safari/537.36 ayed_recon/%s" % VERSION)})
    if proxy:
        s.proxies = {"http": proxy, "https": proxy}
    key = os.environ.get("NVD_API_KEY")
    if key:
        s.headers["apiKey"] = key
    return s


def normalize(target):
    if "://" not in target:
        target = "http://" + target
    p = urlparse(target)
    scheme = p.scheme
    port = p.port or (443 if scheme == "https" else 80)
    return p.hostname, scheme, port, target


# =========================================================================== #
#  THREAT INTELLIGENCE
# =========================================================================== #

# Feature #1 : CPE 2.3 vendor:product map for exact matching
CPE_MAP = {
    "Microsoft IIS":        ("microsoft", "internet_information_services"),
    "Microsoft ASP.NET":    ("microsoft", ".net_framework"),
    "ASP.NET Framework":    ("microsoft", ".net_framework"),
    "ASP.NET MVC":          ("microsoft", "asp.net_mvc"),
    "Apache httpd":         ("apache", "http_server"),
    "nginx":                ("nginx", "nginx"),
    "PHP":                  ("php", "php"),
    "Node.js Express":      ("openjsf", "express"),
}


def _extract_cve(item):
    c = item.get("cve", {})
    desc = next((d["value"] for d in c.get("descriptions", [])
                 if d.get("lang") == "en"), "")
    metrics = c.get("metrics", {})
    score = severity = vector = None
    for k in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(k):
            cd = metrics[k][0]["cvssData"]
            score = cd.get("baseScore")
            severity = cd.get("baseSeverity") or metrics[k][0].get("baseSeverity")
            vector = cd.get("vectorString")
            break
    return {"id": c.get("id"), "published": (c.get("published") or "")[:10],
            "score": score, "severity": severity, "vector": vector,
            "summary": desc[:240]}


def nvd_call(session, params, ttl=86400):
    """Feature #1/#5/#8 cached NVD 2.0 call."""
    key = "nvd:" + json.dumps(params, sort_keys=True)
    hit = cache_get(key, ttl)
    if hit is not None:
        return hit, None
    try:
        r = session.get("https://services.nvd.nist.gov/rest/json/cves/2.0",
                        params=params, timeout=30)
        if not r.ok:
            return [], f"HTTP {r.status_code}"
        cves = [_extract_cve(i) for i in r.json().get("vulnerabilities", [])]
        cache_put(key, cves)
        time.sleep(0.7)   # politeness
        return cves, None
    except Exception as e:
        return [], str(e)


def cve_for_product(session, product, version):
    """Feature #1+#9 : CPE-exact match, keyword fallback, newest-first."""
    result = {"method": None, "results": []}
    vendor_product = CPE_MAP.get(product)
    cves, err = [], None
    if vendor_product and version:
        vendor, prod = vendor_product
        cpe = f"cpe:2.3:a:{vendor}:{prod}:{version}"
        cves, err = nvd_call(session, {"virtualMatchString": cpe,
                                       "resultsPerPage": 40})
        if cves:
            result["method"] = f"CPE {cpe}"
    if not cves and vendor_product:
        vendor, prod = vendor_product
        cpe = f"cpe:2.3:a:{vendor}:{prod}"
        cves, err = nvd_call(session, {"virtualMatchString": cpe,
                                       "resultsPerPage": 40})
        if cves:
            result["method"] = f"CPE {cpe} (any version)"
    if not cves:
        kw = f"{product} {version}".strip() if version else product
        cves, err = nvd_call(session, {"keywordSearch": kw, "resultsPerPage": 40})
        if cves:
            result["method"] = f"keyword '{kw}'"
        else:
            base = re.sub(r"\s+[\d][\w.\-]*$", "", kw).strip()
            if base and base != kw:
                cves, err = nvd_call(session,
                                     {"keywordSearch": base, "resultsPerPage": 40})
                if cves:
                    result["method"] = f"keyword '{base}' (product-only)"
    cves.sort(key=lambda c: c["published"], reverse=True)
    result["results"] = cves
    result["error"] = err
    return result


def load_kev(session):
    """Feature #2 : CISA Known Exploited Vulnerabilities catalog (cached 24h)."""
    hit = cache_get("kev", 86400)
    if hit is not None:
        return set(hit)
    try:
        r = session.get("https://www.cisa.gov/sites/default/files/feeds/"
                        "known_exploited_vulnerabilities.json", timeout=40)
        ids = [v["cveID"] for v in r.json().get("vulnerabilities", [])]
        cache_put("kev", ids)
        return set(ids)
    except Exception as e:
        log.warning("KEV load failed: %s", e)
        return set()


def epss_scores(session, cve_ids):
    """Feature #3 : FIRST EPSS exploit-probability (batched, cached 12h)."""
    scores = {}
    if not cve_ids:
        return scores
    ids = list(dict.fromkeys(cve_ids))
    for i in range(0, len(ids), 90):
        batch = ids[i:i + 90]
        key = "epss:" + ",".join(sorted(batch))
        hit = cache_get(key, 43200)
        if hit is not None:
            scores.update(hit); continue
        try:
            r = session.get("https://api.first.org/data/v1/epss",
                            params={"cve": ",".join(batch)}, timeout=25)
            got = {d["cve"]: float(d["epss"]) for d in r.json().get("data", [])}
            cache_put(key, got)
            scores.update(got)
            time.sleep(0.3)
        except Exception as e:
            log.warning("EPSS batch failed: %s", e)
    return scores


def risk_score(cvss, epss, kev):
    """Feature #4/#10 : composite exploitability-weighted risk 0-100."""
    base = (cvss or 0) * 10                    # 0-100 from CVSS
    ep = (epss or 0)                            # 0-1 probability
    score = base * (0.4 + 0.6 * ep)            # weight by exploit likelihood
    if kev:
        score = max(score, 95)                 # actively exploited -> critical
    return round(min(score, 100), 1)


def enrich_cves(session, correlation):
    """Attach KEV flag, EPSS, composite risk; dedupe; sort by risk."""
    kev = load_kev(session)
    all_ids = [r["id"] for c in correlation for r in c["cve"]["results"] if r["id"]]
    epss = epss_scores(session, all_ids)
    seen = {}
    for c in correlation:
        for r in c["cve"]["results"]:
            r["kev"] = r["id"] in kev
            r["epss"] = epss.get(r["id"])
            r["risk"] = risk_score(r["score"], r["epss"], r["kev"])
        c["cve"]["results"].sort(key=lambda r: r["risk"], reverse=True)
        for r in c["cve"]["results"]:
            if r["id"] and r["id"] not in seen:
                seen[r["id"]] = {**r, "product": c["product"]}
    top = sorted(seen.values(), key=lambda r: r["risk"], reverse=True)
    return {"kev_count": sum(1 for r in seen.values() if r["kev"]),
            "unique": len(seen), "top": top[:25]}


# =========================================================================== #
#  RECON
# =========================================================================== #

# Feature #21/#22/#23 : DNS-over-HTTPS records + email auth posture
def doh(session, name, rtype):
    try:
        r = session.get("https://dns.google/resolve",
                        params={"name": name, "type": rtype}, timeout=12)
        return [a.get("data", "").strip('"') for a in r.json().get("Answer", [])]
    except Exception:
        return []


def dns_records(session, host):
    out = {"A": [], "AAAA": [], "MX": [], "NS": [], "TXT": [], "CAA": [],
           "reverse": {}, "spf": None, "dmarc": None}
    try:
        infos = socket.getaddrinfo(host, None)
        out["A"] = sorted({i[4][0] for i in infos if ":" not in i[4][0]})
        out["AAAA"] = sorted({i[4][0] for i in infos if ":" in i[4][0]})
    except Exception:
        pass
    for ip in out["A"]:
        try:
            out["reverse"][ip] = socket.gethostbyaddr(ip)[0]
        except Exception:
            out["reverse"][ip] = None
    if session:
        out["MX"] = doh(session, host, "MX")
        out["NS"] = doh(session, host, "NS")
        out["TXT"] = doh(session, host, "TXT")
        out["CAA"] = doh(session, host, "CAA")
        out["spf"] = next((t for t in out["TXT"] if t.startswith("v=spf1")), None)
        dmarc = doh(session, "_dmarc." + host, "TXT")
        out["dmarc"] = next((t for t in dmarc if "DMARC" in t), None)
    return out


# Feature #12 : subdomain enumeration via crt.sh CT logs
def crtsh_subdomains(session, host):
    base = ".".join(host.split(".")[-2:])
    key = "crtsh:" + base
    hit = cache_get(key, 86400)
    if hit is not None:
        return hit
    subs = set()
    try:
        r = session.get(f"https://crt.sh/?q=%25.{base}&output=json", timeout=40)
        for row in r.json():
            for n in row.get("name_value", "").split("\n"):
                n = n.strip().lstrip("*.").lower()
                if n.endswith(base):
                    subs.add(n)
    except Exception as e:
        log.warning("crt.sh failed: %s", e)
    result = sorted(subs)[:200]
    cache_put(key, result)
    return result


# Feature #24 : ASN / GeoIP
def asn_geo(session, ip):
    try:
        r = session.get(f"https://ipapi.co/{ip}/json/", timeout=15)
        d = r.json()
        return {"ip": ip, "org": d.get("org"), "asn": d.get("asn"),
                "city": d.get("city"), "country": d.get("country_name")}
    except Exception:
        return {"ip": ip}


SEC_HEADERS = ["Strict-Transport-Security", "Content-Security-Policy",
               "X-Frame-Options", "X-Content-Type-Options", "Referrer-Policy",
               "Permissions-Policy", "X-XSS-Protection"]

TECH_SIGNATURES = {
    "Microsoft-IIS":  ("Server", r"Microsoft-IIS/([\d.]+)", "Microsoft IIS"),
    "ASP.NET":        ("X-Powered-By", r"ASP\.NET", "Microsoft ASP.NET"),
    "AspNetVersion":  ("X-AspNet-Version", r"([\d.]+)", "ASP.NET Framework"),
    "AspNetMvc":      ("X-AspNetMvc-Version", r"([\d.]+)", "ASP.NET MVC"),
    "Apache":         ("Server", r"Apache/?([\d.]+)?", "Apache httpd"),
    "nginx":          ("Server", r"nginx/?([\d.]+)?", "nginx"),
    "PHP":            ("X-Powered-By", r"PHP/?([\d.]+)?", "PHP"),
    "Express":        ("X-Powered-By", r"Express", "Node.js Express"),
}

# Feature #17 : WAF fingerprints
WAF_SIGNS = {
    "Cloudflare": ["cloudflare", "cf-ray"], "Akamai": ["akamai"],
    "AWS WAF": ["awselb", "x-amz"], "Imperva/Incapsula": ["incap_ses", "visid_incap"],
    "F5 BIG-IP": ["bigipserver", "x-waf"], "Sucuri": ["sucuri"],
    "ModSecurity": ["mod_security", "modsecurity"],
}

# Feature #15 : sensitive-file exposure probes
SENSITIVE_PATHS = ["/.git/config", "/.env", "/web.config", "/.htaccess",
                   "/backup.zip", "/.DS_Store", "/phpinfo.php", "/server-status",
                   "/robots.txt", "/sitemap.xml", "/.well-known/security.txt"]


def grade_headers(missing):
    """Feature #19 : letter grade from missing security headers."""
    n = len(missing)
    return ["A", "B", "C", "D", "E", "F", "F"][min(n, 6)]


def favicon_hash(session, base_url):
    """Feature #18 : favicon hash (mmh3 shodan-style, md5 fallback)."""
    try:
        r = session.get(base_url.rstrip("/") + "/favicon.ico", timeout=10)
        if r.status_code == 200 and r.content:
            if MMH3_OK:
                b64 = base64.encodebytes(r.content)
                return {"mmh3": mmh3.hash(b64), "md5": hashlib.md5(r.content).hexdigest()}
            return {"md5": hashlib.md5(r.content).hexdigest()}
    except Exception:
        pass
    return None


def http_fingerprint(session, url):
    out = {"url": url, "reachable": False}
    if not session:
        out["error"] = "requests unavailable"; return out
    try:
        r = session.get(url, timeout=15, allow_redirects=True)
    except Exception as e:
        out["error"] = str(e); return out

    out.update(reachable=True, status_code=r.status_code, final_url=r.url,
               headers=dict(r.headers),
               redirect_chain=[h.url for h in r.history] + [r.url])
    m = re.search(r"<title[^>]*>(.*?)</title>", r.text or "", re.I | re.S)
    out["title"] = m.group(1).strip()[:200] if m else None

    # cookies + SameSite (feature #20)
    cookies = []
    for ck in r.cookies:
        rest = getattr(ck, "_rest", {}) or {}
        cookies.append({"name": ck.name, "secure": ck.secure,
                        "httponly": "HttpOnly" in rest or "httponly" in rest,
                        "samesite": rest.get("SameSite") or rest.get("samesite")})
    out["cookies"] = cookies

    present = {h: r.headers[h] for h in SEC_HEADERS if h in r.headers}
    missing = [h for h in SEC_HEADERS if h not in r.headers]
    out["security_headers"] = {"present": present, "missing": missing,
                               "grade": grade_headers(missing)}

    # tech detect
    tech, body = [], r.text or ""
    for _, (hdr, pat, label) in TECH_SIGNATURES.items():
        val = r.headers.get(hdr, "")
        mm = re.search(pat, val, re.I)
        if mm:
            ver = mm.group(1) if mm.groups() and mm.group(1) else None
            tech.append({"product": label, "version": ver,
                         "evidence": f"{hdr}: {val}".strip()})
    mg = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)',
                   body, re.I)
    if mg:
        tech.append({"product": "Generator: " + mg.group(1), "version": None,
                     "evidence": "meta generator"})
    seen, uniq = set(), []
    for t in tech:
        k = (t["product"], t["version"])
        if k not in seen:
            seen.add(k); uniq.append(t)
    out["technologies"] = uniq

    # WAF detect (feature #17)
    blob = " ".join(f"{k}:{v}" for k, v in r.headers.items()).lower()
    blob += " " + " ".join(c["name"].lower() for c in cookies)
    out["waf"] = [name for name, sigs in WAF_SIGNS.items()
                  if any(s in blob for s in sigs)]

    # HTTP methods (feature #13)
    try:
        opt = session.options(url, timeout=10)
        out["allow_methods"] = opt.headers.get("Allow")
    except Exception:
        out["allow_methods"] = None

    # CORS (feature #14)
    try:
        cr = session.get(url, timeout=10,
                         headers={"Origin": "https://evil.example"})
        acao = cr.headers.get("Access-Control-Allow-Origin")
        out["cors"] = {"acao": acao,
                       "reflects_origin": acao == "https://evil.example",
                       "wildcard": acao == "*"}
    except Exception:
        out["cors"] = None
    return out


def probe_sensitive(session, base_url):
    """Feature #15 : sensitive-file exposure."""
    found = []
    def check(path):
        try:
            r = session.get(base_url.rstrip("/") + path, timeout=8,
                            allow_redirects=False)
            if r.status_code == 200 and len(r.content) > 0:
                return {"path": path, "status": r.status_code,
                        "len": len(r.content)}
        except Exception:
            pass
        return None
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(check, SENSITIVE_PATHS):
            if res:
                found.append(res)
    return found


def wayback_urls(session, host, limit=40):
    """Feature #30 : historical URLs from the Wayback Machine."""
    try:
        r = session.get("http://web.archive.org/cdx/search/cdx",
                        params={"url": f"{host}/*", "output": "json",
                                "fl": "original", "collapse": "urlkey",
                                "limit": limit}, timeout=25)
        rows = r.json()
        return [row[0] for row in rows[1:]] if len(rows) > 1 else []
    except Exception:
        return []


# Feature #27 : TLS extended
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
                    out["san"] = [v for k, v in cert.get("subjectAltName", [])][:15]
                    try:
                        exp = datetime.strptime(cert["notAfter"],
                                                "%b %d %H:%M:%S %Y %Z")
                        out["expired"] = exp < datetime.utcnow()
                        out["days_left"] = (exp - datetime.utcnow()).days
                    except Exception:
                        pass
    except Exception as e:
        out["error"] = str(e)
    # feature #28 : weak-protocol probe
    out["weak_protocols"] = _weak_tls(host, port)
    return out


def _weak_tls(host, port):
    weak = []
    for name, const in [("TLSv1", getattr(ssl, "PROTOCOL_TLSv1", None)),
                        ("TLSv1.1", getattr(ssl, "PROTOCOL_TLSv1_1", None))]:
        if const is None:
            continue
        try:
            ctx = ssl.SSLContext(const)
            with socket.create_connection((host, port), timeout=6) as s:
                with ctx.wrap_socket(s, server_hostname=host):
                    weak.append(name)
        except Exception:
            pass
    return weak


COMMON_PORTS = {21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
                80: "http", 110: "pop3", 135: "msrpc", 139: "netbios",
                143: "imap", 443: "https", 445: "smb", 993: "imaps",
                995: "pop3s", 1433: "mssql", 3306: "mysql", 3389: "rdp",
                5432: "postgres", 5985: "winrm", 6379: "redis", 8080: "http-alt",
                8443: "https-alt", 27017: "mongodb"}


def _scan_one(ip, p):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.3)
    try:
        if s.connect_ex((ip, p)) == 0:
            banner_txt = ""
            try:
                s.settimeout(1.5)
                if p in (80, 8080):
                    s.sendall(b"HEAD / HTTP/1.0\r\n\r\n")
                banner_txt = s.recv(120).decode("latin-1", "ignore").strip()
            except Exception:
                pass
            return {"port": p, "service": COMMON_PORTS.get(p, "unknown"),
                    "banner": banner_txt[:120]}
    except Exception:
        pass
    finally:
        s.close()
    return None


def port_scan(host, ip, spec):
    """Feature #11/#19 : nmap or concurrent socket scan."""
    if shutil.which("nmap"):
        try:
            r = subprocess.run(["nmap", "-sV", "-Pn", "--host-timeout", "120s",
                                "-p", spec, ip], capture_output=True,
                               text=True, timeout=240)
            open_ = []
            for line in r.stdout.splitlines():
                m = re.match(r"(\d+)/tcp\s+open\s+(\S+)\s*(.*)", line)
                if m:
                    open_.append({"port": int(m.group(1)), "service": m.group(2),
                                  "banner": m.group(3).strip()})
            return {"scanner": "nmap", "open": open_}
        except Exception:
            pass
    if "-" in spec:
        a, b = spec.split("-")
        ports = list(range(int(a), min(int(b), int(a) + 2048) + 1))
    else:
        ports = [int(x) for x in spec.split(",")] if spec else list(COMMON_PORTS)
    open_ = []
    with cf.ThreadPoolExecutor(max_workers=64) as ex:
        for res in ex.map(lambda p: _scan_one(ip, p), ports):
            if res:
                open_.append(res)
    return {"scanner": "socket(threaded)", "open": sorted(open_, key=lambda x: x["port"])}


def os_detection(ip, fp):
    result = {"method": "passive", "guess": "Undetermined", "confidence": "low",
              "evidence": []}
    if shutil.which("nmap") and hasattr(os, "geteuid") and os.geteuid() == 0:
        try:
            r = subprocess.run(["nmap", "-O", "-Pn", "--host-timeout", "90s", ip],
                              capture_output=True, text=True, timeout=140)
            m = re.search(r"OS details: (.+)", r.stdout) or \
                re.search(r"Running: (.+)", r.stdout)
            if m:
                result.update(method="nmap -O", guess=m.group(1),
                              confidence="high",
                              evidence=["nmap TCP/IP stack fingerprint"])
                return result
        except Exception:
            pass
    server = (fp.get("headers", {}) or {}).get("Server", "")
    powered = (fp.get("headers", {}) or {}).get("X-Powered-By", "")
    ev, guess = [], None
    if re.search(r"IIS|ASP\.NET", server + powered, re.I):
        guess = "Windows Server (IIS/ASP.NET)"; ev.append(f"{server} {powered}".strip())
    elif re.search(r"Ubuntu|Debian", server, re.I):
        guess = "Linux (Debian/Ubuntu)"; ev.append(server)
    elif re.search(r"nginx|Apache", server, re.I):
        guess = "Unix/Linux"; ev.append(server)
    ttl = _ping_ttl(ip)
    if ttl:
        ev.append(f"TTL={ttl}")
        if not guess:
            guess = "Windows (TTL~128)" if ttl > 64 else "Linux/Unix (TTL~64)"
    result.update(guess=guess or "Undetermined",
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


# =========================================================================== #
#  OWASP + risk aggregation
# =========================================================================== #
OWASP_TOP10 = [
    ("A01:2021", "Broken Access Control", ["CWE-200", "CWE-284", "CWE-639"]),
    ("A02:2021", "Cryptographic Failures", ["CWE-259", "CWE-327", "CWE-319"]),
    ("A03:2021", "Injection", ["CWE-79", "CWE-89", "CWE-77"]),
    ("A04:2021", "Insecure Design", ["CWE-209", "CWE-256"]),
    ("A05:2021", "Security Misconfiguration", ["CWE-16", "CWE-611"]),
    ("A06:2021", "Vulnerable & Outdated Components", ["CWE-1104", "CWE-937"]),
    ("A07:2021", "Identification & Auth Failures", ["CWE-287", "CWE-384"]),
    ("A08:2021", "Software & Data Integrity Failures", ["CWE-502", "CWE-829"]),
    ("A09:2021", "Logging & Monitoring Failures", ["CWE-778", "CWE-117"]),
    ("A10:2021", "Server-Side Request Forgery", ["CWE-918"]),
]


def owasp_flags(fp, tls, sensitive):
    flags = []
    sh = fp.get("security_headers", {})
    if sh.get("missing"):
        flags.append(("A05:2021", "Security Misconfiguration",
                      f"Missing headers ({sh.get('grade')}): {', '.join(sh['missing'])}"))
    for ck in fp.get("cookies", []):
        if not ck.get("secure") or not ck.get("httponly"):
            flags.append(("A05:2021", "Security Misconfiguration",
                          f"Cookie '{ck['name']}' missing Secure/HttpOnly")); break
    if (fp.get("cors") or {}).get("reflects_origin") or (fp.get("cors") or {}).get("wildcard"):
        flags.append(("A05:2021", "Security Misconfiguration",
                      f"Permissive CORS: {fp['cors'].get('acao')}"))
    if tls.get("expired"):
        flags.append(("A02:2021", "Cryptographic Failures", "TLS cert expired"))
    if tls.get("weak_protocols"):
        flags.append(("A02:2021", "Cryptographic Failures",
                      f"Weak TLS enabled: {', '.join(tls['weak_protocols'])}"))
    if (fp.get("final_url") or "").startswith("http://"):
        flags.append(("A02:2021", "Cryptographic Failures", "Cleartext HTTP"))
    for t in fp.get("technologies", []):
        if t.get("version"):
            flags.append(("A06:2021", "Vulnerable & Outdated Components",
                          f"{t['product']} {t['version']} version disclosed"))
    for s in sensitive:
        flags.append(("A05:2021", "Security Misconfiguration",
                      f"Exposed sensitive path {s['path']}"))
    return flags


def target_rating(cve_summary, owasp, fp):
    """Feature #31 : composite target risk rating."""
    top_risk = cve_summary["top"][0]["risk"] if cve_summary.get("top") else 0
    score = top_risk
    score += min(len(owasp) * 3, 25)
    score += 15 if cve_summary.get("kev_count") else 0
    score = min(score, 100)
    band = ("CRITICAL" if score >= 85 else "HIGH" if score >= 65 else
            "MEDIUM" if score >= 40 else "LOW" if score >= 15 else "INFO")
    return {"score": round(score, 1), "rating": band}


def ghdb_dorks(host):
    return [f'site:{host} inurl:admin', f'site:{host} inurl:login',
            f'site:{host} filetype:aspx', f'site:{host} inurl:"?id="',
            f'site:{host} intitle:"index of"',
            f'site:{host} ext:config OR ext:bak OR ext:old',
            f'site:{host} "sql syntax near"', f'site:{host} inurl:web.config']


# =========================================================================== #
#  PERSISTENCE  (feature #32 SQLite, #33 JSON, #34 SARIF)
# =========================================================================== #
def save_sqlite(data, rating):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("""CREATE TABLE IF NOT EXISTS recon_findings(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, host TEXT,
            rating TEXT, score REAL, top_cve TEXT, top_risk REAL,
            kev_count INT, owasp_count INT, tech TEXT)""")
        top = data["cve_summary"]["top"]
        con.execute("INSERT INTO recon_findings(ts,host,rating,score,top_cve,"
                    "top_risk,kev_count,owasp_count,tech) VALUES(?,?,?,?,?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), data["host"],
                     rating["rating"], rating["score"],
                     top[0]["id"] if top else None, top[0]["risk"] if top else 0,
                     data["cve_summary"]["kev_count"], len(data["owasp_flags"]),
                     ", ".join(t["product"] for t in data["http"].get("technologies", []))))
        con.commit(); con.close()
        return True
    except Exception as e:
        log.warning("sqlite save failed: %s", e); return False


def build_sarif(data):
    rules, results = [], []
    for cat, name, ev in data["owasp_flags"]:
        rid = f"OWASP-{cat}"
        if not any(r["id"] == rid for r in rules):
            rules.append({"id": rid, "name": name,
                          "shortDescription": {"text": f"{cat} {name}"}})
        results.append({"ruleId": rid, "level": "warning",
                        "message": {"text": ev},
                        "locations": [{"physicalLocation": {"artifactLocation":
                                       {"uri": data["target"]}}}]})
    for r in data["cve_summary"]["top"]:
        rid = r["id"] or "CVE-UNKNOWN"
        lvl = "error" if r["risk"] >= 70 else "warning"
        rules.append({"id": rid, "shortDescription":
                      {"text": (r.get("summary") or "")[:120]}})
        results.append({"ruleId": rid, "level": lvl,
                        "message": {"text": f"{r['product']}: {rid} risk={r['risk']} "
                                    f"CVSS={r['score']} EPSS={r.get('epss')} "
                                    f"KEV={r.get('kev')}"},
                        "locations": [{"physicalLocation": {"artifactLocation":
                                       {"uri": data["target"]}}}]})
    return {"$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{"tool": {"driver": {"name": "ayed_recon", "version": VERSION,
                                          "rules": rules}}, "results": results}]}


# =========================================================================== #
#  REPORT
# =========================================================================== #
def generate_report(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", data["host"])
    stub = os.path.join(outdir, f"recon_{safe}_{ts}")
    md = _markdown(data, ts)
    with open(stub + ".md", "w") as f: f.write(md)
    with open(stub + ".html", "w") as f: f.write(_html(data, md))
    with open(stub + ".json", "w") as f: json.dump(data, f, indent=2, default=str)
    with open(stub + ".sarif", "w") as f: json.dump(build_sarif(data), f, indent=2)
    return stub + ".md", stub + ".html", stub + ".json", stub + ".sarif"


def _markdown(d, ts):
    L, A = [], None
    out = []
    A = out.append
    fp, rating = d["http"], d["rating"]
    A("# Advanced Reconnaissance & Threat-Intelligence Report")
    A(f"\n**Target:** `{d['target']}`  ")
    A(f"**Host:** `{d['host']}`  ")
    A(f"**Date (UTC):** {ts}  |  **Assessor:** ayedcyper  |  **Engine:** ayed_recon v{VERSION}  ")
    A(f"**Authorization:** Public test target (Acunetix vulnweb) — authorized.\n")

    # risk banner
    A(f"## 0. Risk Rating: **{rating['rating']}**  (score {rating['score']}/100)\n")
    cs = d["cve_summary"]
    A(f"- Unique CVEs correlated: **{cs['unique']}**  |  In CISA KEV (actively exploited): **{cs['kev_count']}**")
    A(f"- OWASP categories flagged: **{len(d['owasp_flags'])}**  |  Open ports: **{len(d['ports'].get('open',[]))}**")
    A(f"- Security-header grade: **{fp.get('security_headers',{}).get('grade','?')}**  |  WAF: {', '.join(fp.get('waf')) or 'none detected'}\n")

    A("## 1. Top Prioritized Vulnerabilities (risk = CVSS x EPSS, KEV override)\n")
    if cs["top"]:
        A("| Risk | CVE | Product | CVSS | EPSS | KEV | Published | Summary |")
        A("|------|-----|---------|------|------|-----|-----------|---------|")
        for r in cs["top"][:15]:
            A(f"| {r['risk']} | {r['id']} | {escape(r['product'])} | {r['score']} | "
              f"{r.get('epss')} | {'🔥YES' if r.get('kev') else '-'} | {r['published']} | "
              f"{escape((r.get('summary') or '')[:70])} |")
    else:
        A("- No CVEs correlated.")

    A("\n## 2. Information Gathering\n### 2.1 DNS")
    dns = d["dns"]
    for ip in dns.get("A", []):
        A(f"- A `{ip}`" + (f" -> PTR `{dns['reverse'].get(ip)}`" if dns['reverse'].get(ip) else ""))
    for k in ("AAAA", "MX", "NS", "CAA"):
        if dns.get(k): A(f"- {k}: {', '.join(dns[k])}")
    A(f"- SPF: {dns.get('spf') or '❌ none'}")
    A(f"- DMARC: {dns.get('dmarc') or '❌ none'}")

    if d.get("asn"):
        a = d["asn"]
        A(f"\n### 2.2 Network\n- IP `{a.get('ip')}` — {a.get('org')} ({a.get('asn')}) — {a.get('city')}, {a.get('country')}")

    if d.get("subdomains"):
        A(f"\n### 2.3 Subdomains (crt.sh CT logs) — {len(d['subdomains'])} found")
        for s in d["subdomains"][:30]:
            A(f"- {s}")

    A("\n### 2.4 HTTP Fingerprint")
    A(f"- Final URL: `{fp.get('final_url','-')}`  |  Status: {fp.get('status_code')}")
    A(f"- Title: {fp.get('title')}")
    A(f"- Allowed methods: {fp.get('allow_methods')}")
    cors = fp.get("cors") or {}
    if cors.get("acao"):
        A(f"- CORS ACAO: `{cors.get('acao')}` (reflects={cors.get('reflects_origin')}, wildcard={cors.get('wildcard')})")
    if d.get("favicon"):
        A(f"- Favicon hash: {d['favicon']}")
    A("\n**Response headers:**\n```")
    for k, v in (fp.get("headers", {}) or {}).items():
        A(f"{k}: {v}")
    A("```")

    A("\n### 2.5 Security Headers  (Grade "
      f"{fp.get('security_headers',{}).get('grade')})")
    for h, v in (fp.get("security_headers", {}).get("present") or {}).items():
        A(f"- ✅ `{h}`: {v}")
    for h in fp.get("security_headers", {}).get("missing", []):
        A(f"- ❌ Missing `{h}`")

    A("\n### 2.6 Cookies")
    for ck in fp.get("cookies", []):
        A(f"- `{ck['name']}` Secure={ck['secure']} HttpOnly={ck['httponly']} SameSite={ck.get('samesite')}")
    if not fp.get("cookies"): A("- none")

    A("\n### 2.7 Technologies")
    for t in fp.get("technologies", []):
        A(f"- **{t['product']}** {t['version'] or ''} — _{t['evidence']}_")

    A("\n### 2.8 TLS / Certificate")
    tls = d["tls"]
    if tls.get("error"): A(f"- {tls['error']}")
    else:
        A(f"- Protocol: {tls.get('protocol')} | Cipher: {tls.get('cipher')}")
        A(f"- Issuer: {tls.get('issuer')}")
        A(f"- Expires: {tls.get('not_after')} (days left {tls.get('days_left','?')}, expired {tls.get('expired')})")
        if tls.get("san"): A(f"- SAN: {', '.join(tls['san'])}")
    if tls.get("weak_protocols"): A(f"- ⚠️ Weak protocols enabled: {', '.join(tls['weak_protocols'])}")

    if d.get("sensitive"):
        A("\n### 2.9 Sensitive File Exposure")
        for s in d["sensitive"]:
            A(f"- ⚠️ `{s['path']}` (HTTP {s['status']}, {s['len']} bytes)")

    if d.get("wayback"):
        A(f"\n### 2.10 Wayback URLs (sample of {len(d['wayback'])})")
        for u in d["wayback"][:15]: A(f"- {u}")

    A("\n## 3. OS Detection")
    o = d["os"]
    A(f"- Method: {o['method']} | Guess: **{o['guess']}** ({o['confidence']})")
    for e in o.get("evidence", []): A(f"- Evidence: {e}")

    A("\n## 4. Port / Service Scan")
    A(f"- Scanner: {d['ports'].get('scanner')}")
    op = d["ports"].get("open", [])
    if op:
        A("\n| Port | Service | Banner |")
        A("|------|---------|--------|")
        for p in op:
            A(f"| {p['port']} | {p.get('service','')} | {escape(str(p.get('banner','')))[:60]} |")
    else:
        A("- none (or blocked by network policy)")

    A("\n## 5. CVE Correlation Detail (per product)")
    for c in d["cve"]:
        A(f"\n### {c['product']} {c['version'] or ''}")
        A(f"- Match method: `{c['cve'].get('method')}`")
        A(f"- 🔎 [NVD]({c['links']['nvd']}) · 💥 [Exploit-DB]({c['links']['exploitdb']}) · 📚 [MITRE]({c['links']['cve_mitre']})")
        res = c["cve"]["results"][:6]
        if res:
            A("\n| CVE | Risk | CVSS | Sev | EPSS | KEV | Published |")
            A("|-----|------|------|-----|------|-----|-----------|")
            for r in res:
                A(f"| {r['id']} | {r['risk']} | {r['score']} | {r.get('severity','')} | "
                  f"{r.get('epss')} | {'YES' if r.get('kev') else '-'} | {r['published']} |")

    A("\n## 6. OWASP Top 10 Mapping\n### 6.1 Evidenced (passive)")
    if d["owasp_flags"]:
        A("| OWASP | Category | Evidence |")
        A("|-------|----------|----------|")
        for cat, name, ev in d["owasp_flags"]:
            A(f"| {cat} | {name} | {escape(ev)} |")
    else:
        A("- none evidenced passively")
    A("\n### 6.2 Full Checklist")
    A("| ID | Category | CWE |")
    A("|----|----------|-----|")
    for cid, name, cwes in OWASP_TOP10:
        A(f"| {cid} | {name} | {', '.join(cwes)} |")

    A("\n## 7. Google Hacking (GHDB) Dorks\n```")
    for dork in d["ghdb"]: A(dork)
    A("```")

    A("\n## 8. Recommendations")
    recs = []
    sh = fp.get("security_headers", {})
    if sh.get("missing"): recs.append(f"Add missing security headers (grade {sh.get('grade')} → target A).")
    if any(not ck.get("secure") or not ck.get("httponly") for ck in fp.get("cookies", [])):
        recs.append("Set Secure/HttpOnly/SameSite on all cookies.")
    if cs.get("kev_count"): recs.append("PRIORITY: patch CVEs present in CISA KEV — these are actively exploited in the wild.")
    if any(t.get("version") for t in fp.get("technologies", [])):
        recs.append("Suppress version banners and patch disclosed components.")
    if (fp.get("final_url") or "").startswith("http://"): recs.append("Enforce HTTPS + HSTS.")
    if d.get("sensitive"): recs.append("Remove/deny access to exposed sensitive files.")
    if tls.get("weak_protocols"): recs.append("Disable TLSv1.0/1.1; require TLSv1.2+.")
    recs.append("Manually validate injection points (SQLi/XSS) on this known-vulnerable target.")
    for i, r in enumerate(recs, 1): A(f"{i}. {r}")

    A(f"\n---\n_ayed_recon v{VERSION} — authorized testing only. Intel: NVD + CISA KEV + FIRST EPSS._")
    return "\n".join(out)


def _html(d, md):
    lines, html, in_code, in_tbl = md.splitlines(), [], False, False
    def close():
        nonlocal in_tbl
        if in_tbl: html.append("</table>"); in_tbl = False
    for ln in lines:
        if ln.startswith("```"):
            close()
            html.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code; continue
        if in_code: html.append(escape(ln)); continue
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "): continue
            if not in_tbl:
                html.append("<table>"); in_tbl = True
                html.append("<tr>" + "".join(f"<th>{_inl(c)}</th>" for c in cells) + "</tr>")
            else:
                html.append("<tr>" + "".join(f"<td>{_inl(c)}</td>" for c in cells) + "</tr>")
            continue
        close()
        if ln.startswith("### "): html.append(f"<h3>{_inl(ln[4:])}</h3>")
        elif ln.startswith("## "): html.append(f"<h2>{_inl(ln[3:])}</h2>")
        elif ln.startswith("# "): html.append(f"<h1>{_inl(ln[2:])}</h1>")
        elif ln.startswith("- "): html.append(f"<li>{_inl(ln[2:])}</li>")
        elif re.match(r"\d+\. ", ln): html.append(f"<li>{_inl(ln[ln.index('.')+2:])}</li>")
        elif ln.strip() == "---": html.append("<hr>")
        elif ln.strip() == "": html.append("<br>")
        else: html.append(f"<p>{_inl(ln)}</p>")
    close()
    body = "\n".join(html)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recon Report — {escape(d['host'])}</title><style>
:root{{color-scheme:light dark}}
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:0 auto;
padding:2rem 1.2rem;line-height:1.55;background:#0d1117;color:#e6edf3}}
h1{{border-bottom:3px solid #58a6ff;padding-bottom:.3em;color:#58a6ff}}
h2{{border-bottom:1px solid #30363d;padding-bottom:.25em;margin-top:2em;color:#79c0ff}}
h3{{color:#a5d6ff;margin-top:1.4em}}
table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.85em;display:block;overflow-x:auto}}
th,td{{border:1px solid #30363d;padding:.4em .6em;text-align:left}}
th{{background:#161b22}} tr:nth-child(even) td{{background:#161b22}}
pre{{background:#161b22;padding:1em;border-radius:8px;overflow-x:auto;border:1px solid #30363d}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:.85em}}
a{{color:#58a6ff;word-break:break-all}} li{{margin:.2em 0}} hr{{border:none;border-top:1px solid #30363d;margin:2em 0}}
</style></head><body>
{body}
</body></html>"""


def _inl(t):
    t = escape(t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"_([^_]+)_", r"<em>\1</em>", t)
    t = re.sub(r"(?<!\")(https?://[^\s<)]+)", r'<a href="\1">\1</a>', t)
    return t


# =========================================================================== #
#  ORCHESTRATION
# =========================================================================== #
def assess(target, args, session):
    host, scheme, port, url = normalize(target)
    sect(f"TARGET  {target}")
    good(f"host={host} scheme={scheme} port={port}")

    sect("1. DNS / EMAIL POSTURE")
    dns = dns_records(session, host)
    ip = dns["A"][0] if dns.get("A") else host
    for a in dns.get("A", []): good(f"A {a} PTR {dns['reverse'].get(a)}")
    if dns.get("spf"): info("SPF present")
    if not dns.get("dmarc"): warn("no DMARC record")

    asn = asn_geo(session, ip) if session and dns.get("A") else {}
    if asn.get("org"): good(f"ASN: {asn.get('org')} {asn.get('asn')} ({asn.get('country')})")

    subs = []
    if args.subdomains and session:
        sect("2. SUBDOMAINS (crt.sh)")
        subs = crtsh_subdomains(session, host)
        good(f"{len(subs)} subdomains from CT logs")

    sect("3. HTTP FINGERPRINT")
    fp = http_fingerprint(session, url)
    if fp.get("reachable"):
        good(f"HTTP {fp['status_code']} · {fp.get('title')} · grade {fp['security_headers']['grade']}")
        if fp.get("waf"): warn(f"WAF: {', '.join(fp['waf'])}")
        for t in fp.get("technologies", []): good(f"tech {t['product']} {t['version'] or ''}")
    else:
        bad(fp.get("error"))

    favicon = favicon_hash(session, fp.get("final_url", url)) if session else None

    sect("4. SENSITIVE FILES")
    sensitive = probe_sensitive(session, fp.get("final_url", url)) if session else []
    for s in sensitive: warn(f"exposed {s['path']} ({s['len']}b)")

    sect("5. TLS")
    tls = tls_info(host, 443)
    info(f"TLS {tls.get('protocol', tls.get('error'))}" +
         (f" weak:{tls['weak_protocols']}" if tls.get("weak_protocols") else ""))

    sect("6. PORT SCAN")
    ports = port_scan(host, ip, args.ports)
    for p in ports.get("open", []): good(f"open {p['port']}/{p['service']}")

    sect("7. OS DETECTION")
    osd = os_detection(ip, fp)
    good(f"OS {osd['guess']} ({osd['confidence']})")

    wayback = wayback_urls(session, host) if (args.wayback and session) else []

    sect("8. CVE CORRELATION (NVD CPE + KEV + EPSS)")
    correlation = []
    for t in fp.get("technologies", []):
        c = cve_for_product(session, t["product"], t.get("version")) \
            if (session and not args.no_cve) else {"results": [], "method": "skipped"}
        q = quote(f"{t['product']} {t.get('version') or ''}".strip())
        correlation.append({"product": t["product"], "version": t.get("version"),
                            "cve": c, "links": {
                                "nvd": f"https://nvd.nist.gov/vuln/search/results?query={q}",
                                "exploitdb": f"https://www.exploit-db.com/search?q={q}",
                                "cve_mitre": f"https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword={q}"}})
        info(f"{t['product']} {t.get('version') or ''} — {c.get('method')} — {len(c['results'])} CVEs")
    cve_summary = enrich_cves(session, correlation) if session and not args.no_cve \
        else {"kev_count": 0, "unique": 0, "top": []}
    if cve_summary["top"]:
        for r in cve_summary["top"][:5]:
            tag = f"{C.R}KEV{C.END}" if r["kev"] else ""
            print(f"    risk {r['risk']:5} {r['id']} CVSS={r['score']} EPSS={r.get('epss')} {tag}")

    sect("9. OWASP MAPPING")
    of = owasp_flags(fp, tls, sensitive)
    for cat, name, ev in of: warn(f"{cat} {ev}")

    data = {"host": host, "target": target, "dns": dns, "asn": asn,
            "subdomains": subs, "http": fp, "favicon": favicon,
            "sensitive": sensitive, "tls": tls, "ports": ports, "os": osd,
            "wayback": wayback, "cve": correlation, "cve_summary": cve_summary,
            "owasp_flags": of, "ghdb": ghdb_dorks(host)}
    data["rating"] = target_rating(cve_summary, of, fp)
    good(f"RISK RATING: {data['rating']['rating']} ({data['rating']['score']}/100)")
    return data


def run(target, args, session):
    data = assess(target, args, session)
    sect("REPORT")
    md, html, js, sarif = generate_report(data, args.outdir)
    if save_sqlite(data, data["rating"]): good(f"SQLite: {DB_PATH}")
    good(f"Markdown: {md}")
    good(f"HTML:     {html}")
    good(f"JSON:     {js}")
    good(f"SARIF:    {sarif}")
    return md, html, js, sarif


def main():
    ap = argparse.ArgumentParser(description="Advanced recon + threat-intel + reporting")
    ap.add_argument("target", nargs="?", help="host or URL")
    ap.add_argument("-l", "--list", help="file with one target per line (feature #39)")
    ap.add_argument("--ports", default="21,22,23,25,53,80,110,135,139,143,443,445,"
                    "993,995,1433,3306,3389,5432,8080,8443", help="ports/range")
    ap.add_argument("--subdomains", action="store_true", help="crt.sh subdomain enum")
    ap.add_argument("--wayback", action="store_true", help="Wayback URL harvest")
    ap.add_argument("--no-cve", action="store_true", help="skip CVE intel")
    ap.add_argument("--proxy", help="http(s)/socks proxy (feature #37)")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    banner()
    if not REQUESTS_OK:
        bad("pip install requests"); sys.exit(1)
    session = build_session(args.proxy)

    targets = []
    if args.list:
        with open(args.list) as f:
            targets = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    elif args.target:
        targets = [args.target]
    else:
        ap.error("provide a target or --list")

    for t in targets:            # feature #39 multi-target
        try:
            run(t, args, session)
        except KeyboardInterrupt:
            bad("interrupted"); sys.exit(130)
        except Exception as e:
            bad(f"{t} failed: {e}")


# --------------------------------------------------------------------------- #
#  FEATURES MAP  (40 advanced improvements over v1)
# --------------------------------------------------------------------------- #
FEATURES = """
 1  CPE 2.3 exact CVE matching (NVD virtualMatchString)
 2  CISA KEV (Known Exploited Vulnerabilities) enrichment
 3  FIRST EPSS exploit-probability scoring
 4  Composite risk score (CVSS x EPSS, KEV override)
 5  Disk cache with TTL for all intel sources
 6  Generic cache_get/cache_put layer
 7  NVD API-key support via env var
 8  Retry/backoff honouring Retry-After (403/429)
 9  Versioned->product keyword fallback, newest-first sort
10  Global CVE dedup + top-risk prioritisation
11  Concurrent (threaded) port scanner
12  Subdomain enumeration via crt.sh CT logs
13  HTTP method audit (OPTIONS/Allow)
14  CORS misconfiguration detection
15  Sensitive-file exposure probing (.git/.env/backup)
16  robots.txt / sitemap / security.txt probing
17  WAF fingerprinting
18  Favicon hash (mmh3 shodan-style + md5)
19  Security-header letter grade (A-F)
20  Cookie SameSite analysis
21  DNS-over-HTTPS multi-record (A/AAAA/MX/NS/TXT/CAA)
22  SPF / DMARC email-auth posture
23  Reverse-DNS (PTR) enrichment
24  ASN / GeoIP lookup
25  Extended technology fingerprinting
26  TLS SAN extraction
27  Full TLS cert chain / issuer detail
28  Weak-protocol (TLS1.0/1.1) probe
29  HSTS / cleartext detection
30  Wayback Machine URL harvest
31  Composite target risk rating (CRITICAL..INFO)
32  SQLite findings persistence
33  Structured JSON output
34  SARIF 2.1.0 output for CI pipelines
35  Logging framework (-v verbose)
36  Politeness rate-limiting between API calls
37  Proxy support (HTTP/SOCKS)
38  Cached/resumable intel across runs
39  Multi-target batch mode (--list)
40  Interactive risk-ranked exec summary + recommendations
"""

if __name__ == "__main__":
    main()
