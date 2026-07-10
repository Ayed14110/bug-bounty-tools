# Advanced Reconnaissance & Threat-Intelligence Report

**Target:** `testaspnet.vulnweb.com`  
**Host:** `testaspnet.vulnweb.com`  
**Date (UTC):** 20260710-032725  |  **Assessor:** ayedcyper  |  **Engine:** ayed_recon v2.0  
**Authorization:** Public test target (Acunetix vulnweb) — authorized.

## 0. Risk Rating: **CRITICAL**  (score 85.6/100)

- Unique CVEs correlated: **41**  |  In CISA KEV (actively exploited): **0**
- OWASP categories flagged: **6**  |  Open ports: **2**
- Security-header grade: **F**  |  WAF: none detected

## 1. Top Prioritized Vulnerabilities (risk = CVSS x EPSS, KEV override)

| Risk | CVE | Product | CVSS | EPSS | KEV | Published | Summary |
|------|-----|---------|------|------|-----|-----------|---------|
| 67.6 | CVE-2007-0042 | Microsoft ASP.NET | 7.8 | 0.77716 | - | 2007-07-10 | Interpretation conflict in ASP.NET in Microsoft .NET Framework 1.0, 1. |
| 64.6 | CVE-2004-0200 | Microsoft ASP.NET | 9.3 | 0.49024 | - | 2004-09-28 | Buffer overflow in the JPEG (JPG) parsing engine in the Microsoft Grap |
| 58.6 | CVE-2005-2127 | Microsoft ASP.NET | 7.5 | 0.63665 | - | 2005-08-19 | Microsoft Internet Explorer 5.01, 5.5, and 6 allows remote attackers t |
| 58.5 | CVE-2012-0163 | Microsoft ASP.NET | 9.3 | 0.38251 | - | 2012-04-10 | Microsoft .NET Framework 1.0 SP3, 1.1 SP1, 2.0 SP2, 3.5, 3.5.1, 4, and |
| 54.6 | CVE-2002-0369 | Microsoft ASP.NET | 10.0 | 0.24346 | - | 2002-07-26 | Buffer overflow in ASP.NET Worker Process allows remote attackers to c |
| 54.3 | CVE-2007-0041 | Microsoft ASP.NET | 9.3 | 0.30666 | - | 2007-07-10 | The PE Loader service in Microsoft .NET Framework 1.0, 1.1, and 2.0 fo |
| 54.3 | CVE-2007-0043 | Microsoft ASP.NET | 9.3 | 0.30666 | - | 2007-07-10 | The Just In Time (JIT) Compiler service in Microsoft .NET Framework 1. |
| 52.2 | CVE-2009-2501 | Microsoft ASP.NET | 9.3 | 0.26824 | - | 2009-10-14 | Heap-based buffer overflow in GDI+ in Microsoft Internet Explorer 6 SP |
| 51.6 | CVE-2009-0091 | Microsoft ASP.NET | 9.3 | 0.25811 | - | 2009-10-14 | Microsoft .NET Framework 2.0, 2.0 SP1, and 3.5 does not properly enfor |
| 51.5 | CVE-2010-3332 | Microsoft ASP.NET | 6.4 | 0.67481 | - | 2010-09-22 | Microsoft .NET Framework 1.1 SP1, 2.0 SP1 and SP2, 3.5, 3.5 SP1, 3.5.1 |
| 51.2 | CVE-2010-1898 | Microsoft ASP.NET | 9.3 | 0.25033 | - | 2010-08-11 | The Common Language Runtime (CLR) in Microsoft .NET Framework 2.0 SP1, |
| 50.5 | CVE-2012-0015 | Microsoft ASP.NET | 9.3 | 0.23778 | - | 2012-02-14 | Microsoft .NET Framework 2.0 SP2 and 3.5.1 does not properly calculate |
| 50.4 | CVE-2010-3958 | Microsoft ASP.NET | 9.3 | 0.23593 | - | 2011-04-13 | The x86 JIT compiler in Microsoft .NET Framework 2.0 SP2, 3.5 SP1, 3.5 |
| 50.4 | CVE-2009-2500 | Microsoft ASP.NET | 9.3 | 0.23647 | - | 2009-10-14 | Integer overflow in GDI+ in Microsoft Internet Explorer 6 SP1, Windows |
| 50.3 | CVE-2009-3126 | Microsoft ASP.NET | 9.3 | 0.23461 | - | 2009-10-14 | Integer overflow in GDI+ in Microsoft Internet Explorer 6 SP1, Windows |

## 2. Information Gathering
### 2.1 DNS
- A `44.238.29.244` -> PTR `ec2-44-238-29-244.us-west-2.compute.amazonaws.com`
- SPF: v=spf1 ~all
- DMARC: ❌ none

### 2.2 Network
- IP `44.238.29.244` — None (None) — None, None

### 2.4 HTTP Fingerprint
- Final URL: `http://testaspnet.vulnweb.com/`  |  Status: 200
- Title: acublog news
- Allowed methods: OPTIONS, TRACE, GET, HEAD, POST
- Favicon hash: {'md5': '50c42a3edaaa2fa00445ac77f1b1a715'}

**Response headers:**
```
cache-control: private
content-type: text/html; charset=utf-8
server: Microsoft-IIS/8.5
x-aspnet-version: 2.0.50727
set-cookie: ASP.NET_SessionId=agwpt3qx05bi5ojb0vired55; path=/; HttpOnly
x-powered-by: ASP.NET
date: Fri, 10 Jul 2026 03:27:10 GMT
content-length: 13917
```

### 2.5 Security Headers  (Grade F)
- ❌ Missing `Strict-Transport-Security`
- ❌ Missing `Content-Security-Policy`
- ❌ Missing `X-Frame-Options`
- ❌ Missing `X-Content-Type-Options`
- ❌ Missing `Referrer-Policy`
- ❌ Missing `Permissions-Policy`
- ❌ Missing `X-XSS-Protection`

### 2.6 Cookies
- `ASP.NET_SessionId` Secure=False HttpOnly=True SameSite=None

### 2.7 Technologies
- **Microsoft IIS** 8.5 — _Server: Microsoft-IIS/8.5_
- **Microsoft ASP.NET**  — _X-Powered-By: ASP.NET_
- **ASP.NET Framework** 2.0.50727 — _X-AspNet-Version: 2.0.50727_

### 2.8 TLS / Certificate
- Protocol: TLSv1.3 | Cipher: TLS_AES_256_GCM_SHA384
- Issuer: None
- Expires: None (days left ?, expired None)

### 2.9 Sensitive File Exposure
- ⚠️ `/robots.txt` (HTTP 200, 13 bytes)

## 3. OS Detection
- Method: passive | Guess: **Undetermined** (low)

## 4. Port / Service Scan
- Scanner: socket(threaded)

| Port | Service | Banner |
|------|---------|--------|
| 80 | http | HTTP/1.1 426 Upgrade Required
content-length: 16
content-t |
| 443 | https |  |

## 5. CVE Correlation Detail (per product)

### Microsoft IIS 8.5
- Match method: `CPE cpe:2.3:a:microsoft:internet_information_services:8.5`
- 🔎 [NVD](https://nvd.nist.gov/vuln/search/results?query=Microsoft%20IIS%208.5) · 💥 [Exploit-DB](https://www.exploit-db.com/search?q=Microsoft%20IIS%208.5) · 📚 [MITRE](https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=Microsoft%20IIS%208.5)

| CVE | Risk | CVSS | Sev | EPSS | KEV | Published |
|-----|------|------|-----|------|-----|-----------|
| CVE-2014-4078 | 25.9 | 5.1 | MEDIUM | 0.18011 | - | 2014-11-11 |

### Microsoft ASP.NET 
- Match method: `CPE cpe:2.3:a:microsoft:.net_framework (any version)`
- 🔎 [NVD](https://nvd.nist.gov/vuln/search/results?query=Microsoft%20ASP.NET) · 💥 [Exploit-DB](https://www.exploit-db.com/search?q=Microsoft%20ASP.NET) · 📚 [MITRE](https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=Microsoft%20ASP.NET)

| CVE | Risk | CVSS | Sev | EPSS | KEV | Published |
|-----|------|------|-----|------|-----|-----------|
| CVE-2007-0042 | 67.6 | 7.8 | HIGH | 0.77716 | - | 2007-07-10 |
| CVE-2004-0200 | 64.6 | 9.3 | HIGH | 0.49024 | - | 2004-09-28 |
| CVE-2005-2127 | 58.6 | 7.5 | HIGH | 0.63665 | - | 2005-08-19 |
| CVE-2012-0163 | 58.5 | 9.3 | HIGH | 0.38251 | - | 2012-04-10 |
| CVE-2002-0369 | 54.6 | 10.0 | HIGH | 0.24346 | - | 2002-07-26 |
| CVE-2007-0041 | 54.3 | 9.3 | HIGH | 0.30666 | - | 2007-07-10 |

### ASP.NET Framework 2.0.50727
- Match method: `CPE cpe:2.3:a:microsoft:.net_framework:2.0.50727`
- 🔎 [NVD](https://nvd.nist.gov/vuln/search/results?query=ASP.NET%20Framework%202.0.50727) · 💥 [Exploit-DB](https://www.exploit-db.com/search?q=ASP.NET%20Framework%202.0.50727) · 📚 [MITRE](https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=ASP.NET%20Framework%202.0.50727)

| CVE | Risk | CVSS | Sev | EPSS | KEV | Published |
|-----|------|------|-----|------|-----|-----------|
| CVE-2008-5100 | 45.0 | 10.0 | HIGH | 0.08367 | - | 2008-11-17 |

## 6. OWASP Top 10 Mapping
### 6.1 Evidenced (passive)
| OWASP | Category | Evidence |
|-------|----------|----------|
| A05:2021 | Security Misconfiguration | Missing headers (F): Strict-Transport-Security, Content-Security-Policy, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, X-XSS-Protection |
| A05:2021 | Security Misconfiguration | Cookie &#x27;ASP.NET_SessionId&#x27; missing Secure/HttpOnly |
| A02:2021 | Cryptographic Failures | Cleartext HTTP |
| A06:2021 | Vulnerable & Outdated Components | Microsoft IIS 8.5 version disclosed |
| A06:2021 | Vulnerable & Outdated Components | ASP.NET Framework 2.0.50727 version disclosed |
| A05:2021 | Security Misconfiguration | Exposed sensitive path /robots.txt |

### 6.2 Full Checklist
| ID | Category | CWE |
|----|----------|-----|
| A01:2021 | Broken Access Control | CWE-200, CWE-284, CWE-639 |
| A02:2021 | Cryptographic Failures | CWE-259, CWE-327, CWE-319 |
| A03:2021 | Injection | CWE-79, CWE-89, CWE-77 |
| A04:2021 | Insecure Design | CWE-209, CWE-256 |
| A05:2021 | Security Misconfiguration | CWE-16, CWE-611 |
| A06:2021 | Vulnerable & Outdated Components | CWE-1104, CWE-937 |
| A07:2021 | Identification & Auth Failures | CWE-287, CWE-384 |
| A08:2021 | Software & Data Integrity Failures | CWE-502, CWE-829 |
| A09:2021 | Logging & Monitoring Failures | CWE-778, CWE-117 |
| A10:2021 | Server-Side Request Forgery | CWE-918 |

## 7. Google Hacking (GHDB) Dorks
```
site:testaspnet.vulnweb.com inurl:admin
site:testaspnet.vulnweb.com inurl:login
site:testaspnet.vulnweb.com filetype:aspx
site:testaspnet.vulnweb.com inurl:"?id="
site:testaspnet.vulnweb.com intitle:"index of"
site:testaspnet.vulnweb.com ext:config OR ext:bak OR ext:old
site:testaspnet.vulnweb.com "sql syntax near"
site:testaspnet.vulnweb.com inurl:web.config
```

## 8. Deep Analysis  —  Assessment Grade **E**  |  Attack-Surface **8/100**
- Latency: 155.9 ms | HTTP/1.1 | 13917 bytes | encoding: None

### 8.1 Security-Header Deep Analysis
- HSTS: present=False max-age=None includeSubDomains=None preload=None
- CSP: present=False unsafe-inline=None unsafe-eval=None wildcard=None
- Clickjacking protected: False | nosniff: False | Referrer-Policy strong: False
- COOP=None COEP=None CORP=None
- Information disclosure: {'server': 'Microsoft-IIS/8.5', 'x_powered_by': 'ASP.NET', 'x_aspnet_version': '2.0.50727'}

### 8.2 Content & Attack Surface
- Links: 60 | External domains: 1 | Forms: 1 | JS files: 0 | iframes: 0
- Login form: False | Upload form: False | Emails found: 0
  - Form `POST http://testaspnet.vulnweb.com/default.aspx` (5 inputs) NO-CSRF

### 8.3 Certificate Deep Analysis
- Key: RSA 2048-bit (weak=False) | Sig: sha256 (weak=False)
- Subject CN: *.vulnweb.com | Issuer CN: Egress Gateway SDS Issuing CA (production) | self-signed: False | wildcard: False
- Days left: 29 | expired: False | expiring-soon: True | serial: 17ca449e7ee837db7d73bd4dc4b7d266
- SAN: testaspnet.vulnweb.com

### 8.4 DNS Deep Analysis
- DNSSEC: False | Wildcard DNS: False | Multi-A/LB: False | DKIM selectors: []
- Verification tokens: 2

### 8.5 CVE Intelligence
- Network-exploitable: 24 | High-EPSS(>0.5): 3 | Exploit-likely: 3 | Avg age: 17.2 yrs
- Attack-vector distribution: {'N': 24, 'L': 1}

**Severity distribution:**
```
HIGH      | ############################## 24
MEDIUM    | # 1
```

**Remediation priority matrix:**
- Immediate (KEV/high-EPSS): CVE-2007-0042, CVE-2005-2127, CVE-2010-3332
- High: CVE-2004-0200, CVE-2012-0163, CVE-2002-0369, CVE-2007-0041, CVE-2007-0043, CVE-2009-2501, CVE-2009-0091, CVE-2010-1898, CVE-2012-0015, CVE-2010-3958

## 9. Recommendations
1. Add missing security headers (grade F → target A).
2. Set Secure/HttpOnly/SameSite on all cookies.
3. Suppress version banners and patch disclosed components.
4. Enforce HTTPS + HSTS.
5. Remove/deny access to exposed sensitive files.
6. Manually validate injection points (SQLi/XSS) on this known-vulnerable target.

---
_ayed_recon v2.0 — authorized testing only. Intel: NVD + CISA KEV + FIRST EPSS._