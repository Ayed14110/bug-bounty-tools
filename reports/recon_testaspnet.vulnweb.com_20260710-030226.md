# Reconnaissance & Vulnerability Correlation Report

**Target:** `testaspnet.vulnweb.com`  
**Host:** `testaspnet.vulnweb.com`  
**Assessment date (UTC):** 20260710-030226  
**Assessor:** ayedcyper  
**Authorization:** Public test target (Acunetix vulnweb) — authorized.

## 1. Executive Summary

- **Reachable:** True (HTTP 200)
- **Detected stack:** Microsoft IIS 8.5, Microsoft ASP.NET, ASP.NET Framework 2.0.50727
- **OS guess:** Undetermined (low confidence)
- **Missing security headers:** 7
- **Open ports found:** 2
- **OWASP categories flagged (passive):** 5

## 2. Information Gathering

### 2.1 DNS
- `44.238.29.244`  ->  PTR `ec2-44-238-29-244.us-west-2.compute.amazonaws.com`

### 2.2 WHOIS / RDAP
- Domain: `VULNWEB.COM`
- Registered: 2010-06-14T07:50:29Z  |  Expires: 2027-06-14T07:50:29Z
- Nameservers: NS-136.AWSDNS-17.COM, NS-1450.AWSDNS-53.ORG, NS-1588.AWSDNS-06.CO.UK, NS-557.AWSDNS-05.NET
- Status: client transfer prohibited

### 2.3 HTTP Fingerprint
- Final URL: `http://testaspnet.vulnweb.com/`
- Title: acublog news

**Response headers:**

```
cache-control: private
content-type: text/html; charset=utf-8
server: Microsoft-IIS/8.5
x-aspnet-version: 2.0.50727
set-cookie: ASP.NET_SessionId=noizfb55e3ldmtvbsles1445; path=/; HttpOnly
x-powered-by: ASP.NET
date: Fri, 10 Jul 2026 03:01:54 GMT
content-length: 12782
```

### 2.4 Security Header Audit
- ❌ Missing `Strict-Transport-Security`
- ❌ Missing `Content-Security-Policy`
- ❌ Missing `X-Frame-Options`
- ❌ Missing `X-Content-Type-Options`
- ❌ Missing `Referrer-Policy`
- ❌ Missing `Permissions-Policy`
- ❌ Missing `X-XSS-Protection`

### 2.5 Cookies
- `ASP.NET_SessionId` Secure=False HttpOnly=False

### 2.6 Technology Detection
- **Microsoft IIS** 8.5 — _Server: Microsoft-IIS/8.5_
- **Microsoft ASP.NET**  — _X-Powered-By: ASP.NET_
- **ASP.NET Framework** 2.0.50727 — _X-AspNet-Version: 2.0.50727_

### 2.7 TLS / Certificate
- Protocol: TLSv1.3  |  Cipher: TLS_AES_256_GCM_SHA384
- Issuer: None
- Valid: None → None (days left: ?, expired: None)

## 3. OS Detection

- **Method:** passive (headers/TTL)
- **Guess:** Undetermined
- **Confidence:** low

## 4. Port / Service Scan

- Scanner: socket

| Port | Service | Banner |
|------|---------|--------|
| 80 | http | HTTP/1.1 426 Upgrade Required
content-length: 16
content-t |
| 443 | https | HTTP/1.1 426 Upgrade Required
content-length: 16
content-t |

## 5. CVE Correlation (NVD / Exploit-DB / MITRE)


### Microsoft IIS 8.5
- 🔎 NVD search: https://nvd.nist.gov/vuln/search/results?query=Microsoft%20IIS%208.5
- 💥 Exploit-DB: https://www.exploit-db.com/search?q=Microsoft%20IIS%208.5
- 📚 CVE/MITRE: https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=Microsoft%20IIS%208.5

### Microsoft ASP.NET 
- 🔎 NVD search: https://nvd.nist.gov/vuln/search/results?query=Microsoft%20ASP.NET
- 💥 Exploit-DB: https://www.exploit-db.com/search?q=Microsoft%20ASP.NET
- 📚 CVE/MITRE: https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=Microsoft%20ASP.NET

  **Recent CVEs (NVD, newest first):**

  | CVE | Published | CVSS | Summary |
  |-----|-----------|------|---------|
  | CVE-2003-0768 | 2003-09-22 | 6.8 | Microsoft ASP.Net 1.1 allows remote attackers to bypass the Cross-Site Scripting |
  | CVE-2004-0847 | 2004-11-03 | 9.8 | The Microsoft .NET forms authentication capability for ASP.NET allows remote att |
  | CVE-2005-0452 | 2005-02-16 | 4.3 | Multiple cross-site scripting (XSS) vulnerabilities in Microsoft ASP.NET (.Net)  |
  | CVE-2005-1664 | 2005-05-18 | 6.4 | The __VIEWSTATE functionality in Microsoft ASP.NET 1.x allows remote attackers t |
  | CVE-2005-1665 | 2005-05-18 | 5.0 | The __VIEWSTATE functionality in Microsoft ASP.NET 1.x, when not cryptographical |

### ASP.NET Framework 2.0.50727
- 🔎 NVD search: https://nvd.nist.gov/vuln/search/results?query=ASP.NET%20Framework%202.0.50727
- 💥 Exploit-DB: https://www.exploit-db.com/search?q=ASP.NET%20Framework%202.0.50727
- 📚 CVE/MITRE: https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=ASP.NET%20Framework%202.0.50727

## 6. OWASP Top 10 Mapping

### 6.1 Passive findings mapped to OWASP

| OWASP | Category | Evidence |
|-------|----------|----------|
| A05:2021 | Security Misconfiguration | Missing security headers: Strict-Transport-Security, Content-Security-Policy, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, X-XSS-Protection |
| A05:2021 | Security Misconfiguration | Cookie &#x27;ASP.NET_SessionId&#x27; missing Secure/HttpOnly flag |
| A02:2021 | Cryptographic Failures | Site served over cleartext HTTP |
| A06:2021 | Vulnerable & Outdated Components | Microsoft IIS version 8.5 disclosed |
| A06:2021 | Vulnerable & Outdated Components | ASP.NET Framework version 2.0.50727 disclosed |

### 6.2 Full OWASP Top 10 checklist (manual verification recommended)

| ID | Category | Related CWE |
|----|----------|-------------|
| A01:2021 | Broken Access Control | CWE-200, CWE-284, CWE-639 |
| A02:2021 | Cryptographic Failures | CWE-259, CWE-327, CWE-319 |
| A03:2021 | Injection (SQLi/XSS/Cmd) | CWE-79, CWE-89, CWE-77 |
| A04:2021 | Insecure Design | CWE-209, CWE-256, CWE-501 |
| A05:2021 | Security Misconfiguration | CWE-16, CWE-611, CWE-548 |
| A06:2021 | Vulnerable & Outdated Components | CWE-1104, CWE-937 |
| A07:2021 | Identification & Auth Failures | CWE-287, CWE-384, CWE-620 |
| A08:2021 | Software & Data Integrity Failures | CWE-502, CWE-829 |
| A09:2021 | Security Logging & Monitoring Failures | CWE-778, CWE-117 |
| A10:2021 | Server-Side Request Forgery (SSRF) | CWE-918 |

## 7. Google Hacking (GHDB) Dorks

Recommended dorks for deeper OSINT on this target:

```
site:testaspnet.vulnweb.com inurl:admin
site:testaspnet.vulnweb.com inurl:login
site:testaspnet.vulnweb.com filetype:aspx
site:testaspnet.vulnweb.com inurl:"?id="
site:testaspnet.vulnweb.com intitle:"index of"
site:testaspnet.vulnweb.com ext:config OR ext:bak OR ext:old
site:testaspnet.vulnweb.com "sql syntax near" OR "Warning: mysql"
site:testaspnet.vulnweb.com inurl:web.config
```

## 8. Recommendations

1. Add the missing HTTP security headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options).
2. Set `Secure` and `HttpOnly` (and `SameSite`) flags on all session cookies.
3. Suppress version banners (Server, X-AspNet-Version, X-Powered-By) and patch disclosed components.
4. Enforce HTTPS site-wide and redirect HTTP→HTTPS.
5. Manually validate injection points (SQLi/XSS) — this target is known-vulnerable by design.
6. Cross-check each disclosed component version against the linked NVD/Exploit-DB results.

---
_Generated by ayed_recon.py — authorized testing only._