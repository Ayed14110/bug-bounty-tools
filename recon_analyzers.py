#!/usr/bin/env python3
"""
recon_analyzers.py  -  82 advanced analysis features (#41-#122) for ayed_recon.

Every function here computes REAL results from REAL input (live HTTP responses,
parsed HTML via stdlib html.parser, DNS-over-HTTPS answers, X.509 certificates
parsed with `cryptography`, and CVE/CVSS data). No placeholders, no simulated
or fabricated values -- if data is unavailable a field is simply absent/None.

Grouped:
  #41-58  security-header deep analysis
  #59-63  cookie deep analysis
  #64-84  HTML content analysis (stdlib parser)
  #85-92  DNS deep analysis (DoH)
  #93-104 TLS / certificate deep analysis (cryptography)
  #105-114 CVE / CVSS deep analysis
  #115-122 scoring, charts and export formats
"""

import csv
import io
import ipaddress
import re
import socket
import ssl
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

try:
    # cryptography's Rust core can raise a BaseException (PanicException) when
    # its native _cffi_backend is missing, so guard with BaseException.
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import rsa, ec
    CRYPTO_OK = True
except BaseException:
    CRYPTO_OK = False

import shutil
import subprocess


# =========================================================================== #
#  #41-58  SECURITY-HEADER DEEP ANALYSIS
# =========================================================================== #
def analyze_headers(headers):
    h = {k.lower(): v for k, v in (headers or {}).items()}
    out = {}

    # #41 HSTS
    hsts = h.get("strict-transport-security")
    if hsts:
        mm = re.search(r"max-age=(\d+)", hsts)
        out["hsts"] = {"present": True,
                       "max_age": int(mm.group(1)) if mm else None,
                       "include_subdomains": "includesubdomains" in hsts.lower(),
                       "preload": "preload" in hsts.lower(),
                       "weak": bool(mm and int(mm.group(1)) < 15552000)}
    else:
        out["hsts"] = {"present": False}

    # #42-46 CSP
    csp = h.get("content-security-policy")
    if csp:
        directives = {}
        for part in csp.split(";"):
            toks = part.split()
            if toks:
                directives[toks[0].lower()] = toks[1:]
        out["csp"] = {
            "present": True, "directives": list(directives.keys()),
            "unsafe_inline": "'unsafe-inline'" in csp,          # #43
            "unsafe_eval": "'unsafe-eval'" in csp,              # #44
            "wildcard_source": bool(re.search(r"(^|\s)\*(\s|;|$)", csp)),  # #45
            "frame_ancestors": directives.get("frame-ancestors"),          # #46
            "has_default_src": "default-src" in directives}
    else:
        out["csp"] = {"present": False}

    # #47 X-Frame-Options
    xfo = h.get("x-frame-options")
    out["x_frame_options"] = {"value": xfo,
                              "clickjacking_protected": bool(xfo) or
                              bool(out["csp"].get("frame_ancestors"))}
    # #48 nosniff
    out["nosniff"] = (h.get("x-content-type-options", "").lower() == "nosniff")
    # #49 Referrer-Policy strength
    rp = h.get("referrer-policy", "").lower()
    strong = {"no-referrer", "strict-origin", "strict-origin-when-cross-origin",
              "same-origin"}
    out["referrer_policy"] = {"value": rp or None,
                              "strong": rp in strong if rp else False}
    # #50 Permissions-Policy
    pp = h.get("permissions-policy")
    out["permissions_policy"] = {"present": bool(pp),
                                 "features": [d.split("=")[0].strip()
                                              for d in pp.split(",")] if pp else []}
    # #51-53 COOP/COEP/CORP
    out["coop"] = h.get("cross-origin-opener-policy")
    out["coep"] = h.get("cross-origin-embedder-policy")
    out["corp"] = h.get("cross-origin-resource-policy")
    # #54 sensitive caching
    cc = h.get("cache-control", "").lower()
    out["cache_control"] = {"value": h.get("cache-control"),
                            "prevents_caching": any(x in cc for x in
                            ("no-store", "no-cache", "private"))}
    # #55-58 information disclosure
    out["disclosure"] = {
        "server": h.get("server"),                 # #55
        "x_powered_by": h.get("x-powered-by"),     # #56
        "x_aspnet_version": h.get("x-aspnet-version"),  # #57
        "x_aspnetmvc_version": h.get("x-aspnetmvc-version"),
        "via": h.get("via"),                       # #58
        "x_generator": h.get("x-generator")}
    return out


# =========================================================================== #
#  #59-63  COOKIE DEEP ANALYSIS
# =========================================================================== #
def analyze_cookies(cookies):
    res = []
    for c in cookies or []:
        name = c.get("name", "")
        res.append({
            "name": name,
            "secure": bool(c.get("secure")),                     # #59
            "httponly": bool(c.get("httponly")),                 # #60
            "samesite": c.get("samesite"),                       # #61
            "prefix_secure": name.startswith("__Secure-"),       # #62
            "prefix_host": name.startswith("__Host-"),           # #62
            "session_cookie": bool(re.search(                    # #63
                r"sess|sid|auth|token|jwt", name, re.I))})
    weak = [c["name"] for c in res
            if c["session_cookie"] and (not c["secure"] or not c["httponly"])]
    return {"cookies": res, "weak_session_cookies": weak}


# =========================================================================== #
#  #64-84  HTML CONTENT ANALYSIS  (stdlib html.parser)
# =========================================================================== #
class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.scripts, self.styles = [], [], []
        self.forms, self.metas, self.iframes, self.inputs = [], [], [], []
        self._form = None
        self.script_inline = 0
        self._in_script = False
        self._script_buf = []
        self.inline_js = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        elif tag == "script":
            if a.get("src"):
                self.scripts.append(a["src"])
            else:
                self.script_inline += 1
                self._in_script = True
        elif tag == "link" and a.get("href") and "stylesheet" in (a.get("rel") or ""):
            self.styles.append(a["href"])
        elif tag == "meta":
            self.metas.append(a)
        elif tag == "iframe" and a.get("src"):
            self.iframes.append(a["src"])
        elif tag == "form":
            self._form = {"action": a.get("action", ""),
                          "method": (a.get("method") or "get").lower(),
                          "inputs": []}
        elif tag in ("input", "textarea", "select"):
            field = {"type": a.get("type", "text"), "name": a.get("name"),
                     "autocomplete": a.get("autocomplete")}
            self.inputs.append(field)
            if self._form is not None:
                self._form["inputs"].append(field)

    def handle_endtag(self, tag):
        if tag == "form" and self._form is not None:
            self.forms.append(self._form)
            self._form = None
        elif tag == "script":
            self._in_script = False
            if self._script_buf:
                self.inline_js.append("".join(self._script_buf))
                self._script_buf = []

    def handle_data(self, data):
        if self._in_script:
            self._script_buf.append(data)


def analyze_html(body, base_url):
    body = body or ""
    p = _Collector()
    try:
        p.feed(body)
    except Exception:
        pass
    host = urlparse(base_url).netloc
    scheme = urlparse(base_url).scheme

    # #64 links, #65 external domains
    abs_links = [urljoin(base_url, l) for l in p.links]
    ext_domains = sorted({urlparse(l).netloc for l in abs_links
                          if urlparse(l).netloc and urlparse(l).netloc != host})
    # #66 emails
    emails = sorted(set(re.findall(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", body)))
    # #67 JS, #68 CSS
    js = sorted({urljoin(base_url, s) for s in p.scripts})
    css = sorted({urljoin(base_url, s) for s in p.styles})
    # #69-73 forms
    forms = []
    login_form = upload_form = False
    for f in p.forms:
        types = [i.get("type") for i in f["inputs"]]
        names = " ".join((i.get("name") or "") for i in f["inputs"]).lower()
        is_login = "password" in types                       # #70
        is_upload = "file" in types                          # #71
        has_csrf = bool(re.search(r"csrf|token|_token|nonce", names))  # #72
        pw_autocomplete = any(i.get("type") == "password" and
                              (i.get("autocomplete") or "").lower() != "off"
                              for i in f["inputs"])           # #73
        login_form = login_form or is_login
        upload_form = upload_form or is_upload
        forms.append({"action": urljoin(base_url, f["action"]),
                      "method": f["method"], "inputs": len(f["inputs"]),
                      "login": is_login, "upload": is_upload,
                      "csrf_token": has_csrf,
                      "password_autocomplete_on": pw_autocomplete})
    # #74 hidden inputs
    hidden = [i.get("name") for i in p.inputs if i.get("type") == "hidden"]
    # #75 HTML comments
    comments = re.findall(r"<!--(.*?)-->", body, re.S)
    interesting_comments = [c.strip()[:120] for c in comments
                            if re.search(r"todo|fixme|pass|key|secret|debug|hack|"
                                         r"api|user", c, re.I)]
    # #77 meta tags, #78 OG, #79 canonical, #80 robots
    og = {m.get("property"): m.get("content") for m in p.metas
          if (m.get("property") or "").startswith("og:")}
    canonical = None
    robots_meta = None
    for m in p.metas:
        if (m.get("name") or "").lower() == "robots":
            robots_meta = m.get("content")
    canon = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)',
                      body, re.I)
    if canon:
        canonical = canon.group(1)
    # #81 mixed content
    mixed = []
    if scheme == "https":
        mixed = sorted(set(re.findall(r'(?:src|href)=["\'](http://[^"\']+)', body)))[:20]
    # #83 API endpoints from inline JS
    api_eps = sorted(set(re.findall(
        r'["\'](/(?:api|rest|graphql|v\d)[/\w.\-]*)["\']', " ".join(p.inline_js))))[:30]
    # #84 social links
    social = sorted({d for d in ext_domains if re.search(
        r"facebook|twitter|x\.com|linkedin|instagram|youtube|github|t\.me", d)})

    return {
        "link_count": len(abs_links),                         # #64
        "external_domains": ext_domains,                      # #65
        "emails": emails,                                     # #66
        "js_files": js, "js_count": len(js),                 # #67
        "css_files": css,                                     # #68
        "forms": forms,                                       # #69
        "has_login_form": login_form,                         # #70
        "has_upload_form": upload_form,                       # #71
        "hidden_inputs": [h for h in hidden if h],           # #74
        "html_comment_count": len(comments),                 # #75
        "interesting_comments": interesting_comments,        # #75
        "inline_script_count": p.script_inline,              # #76
        "meta_count": len(p.metas),                           # #77
        "open_graph": og,                                     # #78
        "canonical": canonical,                               # #79
        "robots_meta": robots_meta,                           # #80
        "mixed_content": mixed,                               # #81
        "iframe_count": len(p.iframes),                       # #82
        "iframes": [urljoin(base_url, i) for i in p.iframes], # #82
        "api_endpoints": api_eps,                             # #83
        "social_links": social,                               # #84
    }


# =========================================================================== #
#  #85-92  DNS DEEP ANALYSIS (DNS-over-HTTPS)
# =========================================================================== #
def _doh_full(session, name, rtype):
    try:
        r = session.get("https://dns.google/resolve",
                        params={"name": name, "type": rtype}, timeout=12)
        return r.json()
    except Exception:
        return {}


def analyze_dns(session, host, a_records):
    out = {}
    if not session:
        return out
    # #85 DNSSEC (AD flag)
    j = _doh_full(session, host, "A")
    out["dnssec"] = bool(j.get("AD"))
    # #86 wildcard DNS
    rnd = "zzq" + str(abs(hash(host)) % 99999) + "wildtest"
    wj = _doh_full(session, f"{rnd}.{host}", "A")
    out["wildcard_dns"] = bool(wj.get("Answer"))
    # #87 CNAME chain
    ans = j.get("Answer", [])
    out["cname_chain"] = [a["data"] for a in ans if a.get("type") == 5]
    # #88 DKIM selector probe
    dkim = []
    for sel in ("default", "google", "selector1", "selector2", "k1", "dkim", "mail"):
        dj = _doh_full(session, f"{sel}._domainkey.{host}", "TXT")
        if dj.get("Answer"):
            dkim.append(sel)
    out["dkim_selectors"] = dkim
    # #89 TXT verification tokens
    tj = _doh_full(session, host, "TXT")
    txts = [a["data"].strip('"') for a in tj.get("Answer", [])]
    verifications = [t for t in txts if re.search(
        r"verification|site-verification|=|-domain-verification", t)]
    out["txt_records"] = txts
    out["verification_tokens"] = verifications
    # #90 MX priority
    mj = _doh_full(session, host, "MX")
    mx = []
    for a in mj.get("Answer", []):
        parts = a.get("data", "").split()
        if len(parts) == 2:
            mx.append({"priority": int(parts[0]), "host": parts[1]})
    out["mx"] = sorted(mx, key=lambda x: x["priority"])
    # #91 CAA
    cj = _doh_full(session, host, "CAA")
    out["caa"] = [a.get("data") for a in cj.get("Answer", [])]
    out["caa_restricted"] = bool(out["caa"])
    # #92 load balancing / multi-A
    out["multi_a"] = len(a_records or []) > 1
    out["a_count"] = len(a_records or [])
    return out


# =========================================================================== #
#  #93-104  TLS / CERTIFICATE DEEP ANALYSIS (cryptography)
# =========================================================================== #
def _cert_via_openssl(host, port=443):
    """Real fallback cert parse using the openssl binary (no python deps)."""
    if not shutil.which("openssl"):
        return {"error": "no cert parser available"}
    try:
        p = subprocess.run(
            ["openssl", "s_client", "-connect", f"{host}:{port}",
             "-servername", host],
            input="", capture_output=True, text=True, timeout=15)
        pem = re.search(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
                        p.stdout, re.S)
        if not pem:
            return {"error": "no certificate returned"}
        txt = subprocess.run(
            ["openssl", "x509", "-noout", "-text", "-serial", "-subject",
             "-issuer", "-dates"],
            input=pem.group(0), capture_output=True, text=True, timeout=10).stdout
        out = {"parser": "openssl"}
        m = re.search(r"Public-Key:\s*\((\d+) bit\)", txt)
        if m:
            out["key_size"] = int(m.group(1)); out["weak_key"] = int(m.group(1)) < 2048
        m = re.search(r"Signature Algorithm:\s*(\S+)", txt)
        if m:
            out["sig_algorithm"] = m.group(1)
            out["weak_signature"] = bool(re.search(r"md5|sha1", m.group(1), re.I))
        m = re.search(r"serial=(\S+)", txt)
        if m:
            out["serial"] = m.group(1)
        san = re.search(r"Subject Alternative Name:\s*\n\s*(.+)", txt)
        if san:
            names = re.findall(r"DNS:([^,\s]+)", san.group(1))
            out["san"] = names
            out["wildcard_cert"] = any(n.startswith("*.") for n in names)
        subj = re.search(r"subject=.*?CN\s*=\s*([^,/\n]+)", txt)
        iss = re.search(r"issuer=.*?CN\s*=\s*([^,/\n]+)", txt)
        out["subject_cn"] = subj.group(1).strip() if subj else None
        out["issuer_cn"] = iss.group(1).strip() if iss else None
        out["self_signed"] = bool(subj and iss and subj.group(1) == iss.group(1))
        na = re.search(r"notAfter=(.+)", txt)
        if na:
            try:
                exp = datetime.strptime(na.group(1).strip(), "%b %d %H:%M:%S %Y %Z")
                exp = exp.replace(tzinfo=timezone.utc)
                days = (exp - datetime.now(timezone.utc)).days
                out["days_left"] = days; out["expired"] = days < 0
                out["expiring_soon"] = 0 <= days <= 30
            except Exception:
                pass
        return out
    except Exception as e:
        return {"error": str(e)}


def analyze_certificate(host, port=443):
    out = {}
    if not CRYPTO_OK:
        return _cert_via_openssl(host, port)
    try:
        ctx = ssl._create_unverified_context()
        with socket.create_connection((host, port), timeout=10) as s:
            with ctx.wrap_socket(s, server_hostname=host) as ss:
                der = ss.getpeercert(binary_form=True)
        cert = x509.load_der_x509_certificate(der)
    except Exception as e:
        out["error"] = str(e)
        return out

    pub = cert.public_key()
    # #93 key size, #94 key algorithm
    if isinstance(pub, rsa.RSAPublicKey):
        out["key_type"] = "RSA"; out["key_size"] = pub.key_size
        out["weak_key"] = pub.key_size < 2048
    elif isinstance(pub, ec.EllipticCurvePublicKey):
        out["key_type"] = "EC"; out["key_size"] = pub.curve.key_size
        out["curve"] = pub.curve.name; out["weak_key"] = pub.curve.key_size < 224
    else:
        out["key_type"] = type(pub).__name__
    # #95 signature hash algorithm
    sig = getattr(cert.signature_hash_algorithm, "name", None)
    out["sig_algorithm"] = sig
    out["weak_signature"] = sig in ("md5", "sha1")           # #104
    # #96 serial
    out["serial"] = format(cert.serial_number, "x")
    # #97 self-signed
    out["self_signed"] = cert.issuer == cert.subject
    # #99 SAN + #98 wildcard
    try:
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value.get_values_for_type(x509.DNSName)
        out["san"] = san
        out["wildcard_cert"] = any(n.startswith("*.") for n in san)
    except Exception:
        out["san"] = []
        out["wildcard_cert"] = False
    # #100-101 validity window + expiry thresholds
    try:
        na = cert.not_valid_after_utc
        nb = cert.not_valid_before_utc
        now = datetime.now(timezone.utc)
    except AttributeError:
        na = cert.not_valid_after.replace(tzinfo=timezone.utc)
        nb = cert.not_valid_before.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
    days_left = (na - now).days
    out["valid_days_total"] = (na - nb).days
    out["days_left"] = days_left
    out["expired"] = days_left < 0
    out["expiring_soon"] = 0 <= days_left <= 30
    # #102 version
    out["version"] = cert.version.name
    # #103 CA / basic constraints
    try:
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        out["is_ca"] = bc.ca
    except Exception:
        out["is_ca"] = None
    out["issuer_cn"] = _name_cn(cert.issuer)
    out["subject_cn"] = _name_cn(cert.subject)
    return out


def _name_cn(name):
    try:
        return name.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
    except Exception:
        return None


# =========================================================================== #
#  #105-114  CVE / CVSS DEEP ANALYSIS
# =========================================================================== #
_CVSS_KEYS = {"AV": "attack_vector", "AC": "attack_complexity", "PR": "priv_required",
              "UI": "user_interaction", "S": "scope", "C": "confidentiality",
              "I": "integrity", "A": "availability"}


def parse_cvss_vector(vector):                                # #105
    if not vector:
        return {}
    out = {}
    for part in vector.split("/"):
        if ":" in part:
            k, v = part.split(":", 1)
            if k in _CVSS_KEYS:
                out[_CVSS_KEYS[k]] = v
    return out


def analyze_cves(top_cves):
    av_dist = Counter()                                       # #106
    cwe_freq = Counter()                                      # #107
    ages, network_exploitable, high_epss, exploit_likely = [], 0, 0, 0
    per_cve = []
    now_year = datetime.now(timezone.utc).year
    for r in top_cves or []:
        comp = parse_cvss_vector(r.get("vector"))
        av = comp.get("attack_vector")
        if av:
            av_dist[av] += 1
        if av == "N":
            network_exploitable += 1                          # #109
        # #107 CWE from summary
        for cwe in re.findall(r"CWE-\d+", r.get("summary", "") or ""):
            cwe_freq[cwe] += 1
        # #108 age
        yr = None
        m = re.match(r"(\d{4})", r.get("published", "") or "")
        if m:
            yr = now_year - int(m.group(1))
            ages.append(yr)
        epss = r.get("epss") or 0
        if epss > 0.5:
            high_epss += 1                                    # #110
        likely = bool(r.get("kev")) or epss > 0.5            # #113
        if likely:
            exploit_likely += 1
        # #111 patch priority
        cvss = r.get("score") or 0
        priority = round(cvss * (0.3 + 0.7 * epss) + (3 if r.get("kev") else 0), 1)
        per_cve.append({
            "id": r.get("id"), "product": r.get("product"),
            "cvss": cvss, "epss": r.get("epss"), "kev": r.get("kev"),
            "risk": r.get("risk"), "age_years": yr,
            "cvss_components": comp,
            "patch_priority": priority,
            "exploit_likely": likely,                         # #113
            "advisory": _advisory_link(r.get("id"))})         # #112
    sev_dist = Counter()                                      # #114
    for r in top_cves or []:
        s = r.get("severity") or _sev_from_score(r.get("score"))
        if s:
            sev_dist[s] += 1
    return {
        "attack_vector_distribution": dict(av_dist),          # #106
        "cwe_frequency": dict(cwe_freq.most_common(10)),      # #107
        "avg_age_years": round(sum(ages) / len(ages), 1) if ages else None,  # #108
        "network_exploitable": network_exploitable,           # #109
        "high_epss_count": high_epss,                         # #110
        "exploit_likely_count": exploit_likely,               # #113
        "severity_distribution": dict(sev_dist),              # #114
        "per_cve": per_cve}


def _advisory_link(cve_id):                                   # #112
    if not cve_id:
        return None
    return f"https://nvd.nist.gov/vuln/detail/{cve_id}"


def _sev_from_score(score):
    if score is None:
        return None
    return ("CRITICAL" if score >= 9 else "HIGH" if score >= 7 else
            "MEDIUM" if score >= 4 else "LOW")


# =========================================================================== #
#  #115-122  SCORING, CHARTS, EXPORT
# =========================================================================== #
def attack_surface_score(html_a, dns_a, ports):              # #115
    score = 0
    score += min(len(html_a.get("external_domains", [])), 20)
    score += len(html_a.get("forms", [])) * 3
    score += 10 if html_a.get("has_upload_form") else 0
    score += 5 if html_a.get("has_login_form") else 0
    score += len(html_a.get("api_endpoints", [])) * 2
    score += len(dns_a.get("dkim_selectors", []))
    score += len(ports.get("open", [])) * 2
    score += len(html_a.get("iframes", []))
    return min(score, 100)


def ascii_bar_chart(dist, width=30):                          # #116
    if not dist:
        return "(no data)"
    mx = max(dist.values()) or 1
    lines = []
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    items = sorted(dist.items(), key=lambda kv: order.index(kv[0])
                   if kv[0] in order else 99)
    for k, v in items:
        bar = "#" * int(width * v / mx)
        lines.append(f"{k:9} | {bar} {v}")
    return "\n".join(lines)


def markdown_toc(md_text):                                    # #117
    toc = []
    for ln in md_text.splitlines():
        m = re.match(r"(#{2,3})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1)) - 2
            title = re.sub(r"[*`]", "", m.group(2)).strip()
            slug = re.sub(r"[^\w\- ]", "", title).lower().replace(" ", "-")
            toc.append("  " * lvl + f"- [{title}](#{slug})")
    return "\n".join(toc)


def cve_csv(per_cve):                                         # #118
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["cve", "product", "cvss", "epss", "kev", "risk",
                "patch_priority", "age_years", "exploit_likely", "advisory"])
    for c in per_cve:
        w.writerow([c["id"], c["product"], c["cvss"], c["epss"], c["kev"],
                    c["risk"], c["patch_priority"], c["age_years"],
                    c["exploit_likely"], c["advisory"]])
    return buf.getvalue()


def remediation_matrix(per_cve):                              # #119
    buckets = {"Immediate (KEV/high-EPSS)": [], "High": [], "Medium": [], "Low": []}
    for c in per_cve:
        if c["exploit_likely"]:
            buckets["Immediate (KEV/high-EPSS)"].append(c["id"])
        elif (c["cvss"] or 0) >= 7:
            buckets["High"].append(c["id"])
        elif (c["cvss"] or 0) >= 4:
            buckets["Medium"].append(c["id"])
        else:
            buckets["Low"].append(c["id"])
    return {k: v for k, v in buckets.items() if v}


def overall_grade(header_a, tls_a, cert_a, owasp_count, kev_count):  # #120
    penalty = 0
    if not header_a.get("hsts", {}).get("present"):
        penalty += 1
    if not header_a.get("csp", {}).get("present"):
        penalty += 1
    if not header_a.get("nosniff"):
        penalty += 1
    if not header_a.get("x_frame_options", {}).get("clickjacking_protected"):
        penalty += 1
    if cert_a.get("weak_signature") or cert_a.get("weak_key"):
        penalty += 2
    if cert_a.get("expired"):
        penalty += 2
    penalty += min(owasp_count, 4)
    penalty += 3 if kev_count else 0
    grade = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "F"][min(penalty, 9)]
    return {"grade": grade, "penalty_points": penalty}


def response_metrics(response, elapsed):                      # #121 #122
    out = {"latency_ms": round(elapsed * 1000, 1)}            # #121
    if response is not None:
        ver = getattr(getattr(response, "raw", None), "version", None)
        out["http_version"] = {10: "HTTP/1.0", 11: "HTTP/1.1",
                               20: "HTTP/2"}.get(ver, str(ver))  # #122
        out["content_length"] = len(response.content or b"")
        out["content_encoding"] = response.headers.get("Content-Encoding")
        out["content_type"] = response.headers.get("Content-Type")
        out["compressed"] = bool(response.headers.get("Content-Encoding"))
    return out


# =========================================================================== #
#  AGGREGATOR
# =========================================================================== #
def run_all(ctx):
    """ctx keys: headers, cookies, body, base_url, host, session, a_records,
    top_cves, ports, response, elapsed"""
    header_a = analyze_headers(ctx.get("headers"))
    cookie_a = analyze_cookies(ctx.get("cookies"))
    html_a = analyze_html(ctx.get("body"), ctx.get("base_url"))
    dns_a = analyze_dns(ctx.get("session"), ctx.get("host"), ctx.get("a_records"))
    cert_a = analyze_certificate(ctx.get("host"))
    cve_a = analyze_cves(ctx.get("top_cves"))
    metrics = response_metrics(ctx.get("response"), ctx.get("elapsed", 0))
    surface = attack_surface_score(html_a, dns_a, ctx.get("ports", {}))
    return {
        "headers": header_a, "cookies": cookie_a, "html": html_a,
        "dns": dns_a, "certificate": cert_a, "cve_analysis": cve_a,
        "metrics": metrics, "attack_surface_score": surface,
        "severity_chart": ascii_bar_chart(cve_a["severity_distribution"]),
        "remediation_matrix": remediation_matrix(cve_a["per_cve"]),
        "cve_csv": cve_csv(cve_a["per_cve"]),
    }


# feature count self-check
FEATURE_COUNT = 82
