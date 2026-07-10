# Reconnaissance & Vulnerability Correlation Report

**Target:** `testaspnet.vulnweb.com`  
**Host:** `testaspnet.vulnweb.com`  
**Assessment date (UTC):** 20260710-030650  
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
set-cookie: ASP.NET_SessionId=utbyua45umtjifz1ujq4iv55; path=/; HttpOnly
x-powered-by: ASP.NET
date: Fri, 10 Jul 2026 03:06:17 GMT
content-length: 13916
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

  **Recent CVEs (NVD keyword `Microsoft IIS (product-only fallback)`, newest first):**

  | CVE | Published | CVSS | Severity | Summary |
  |-----|-----------|------|----------|---------|
  | CVE-2006-1394 | 2006-03-26 | 4.3 | MEDIUM | Multiple cross-site scripting (XSS) vulnerabilities in the Microsoft IIS ISAPI f |
  | CVE-2005-4360 | 2005-12-20 | 7.8 | HIGH | The URL parser in Microsoft Internet Information Services (IIS) 5.1 on Windows X |
  | CVE-2005-2678 | 2005-08-23 | 5.0 | MEDIUM | Microsoft IIS 5.1 and 6 allows remote attackers to spoof the SERVER_NAME variabl |
  | CVE-2005-2089 | 2005-07-05 | 4.3 | MEDIUM | Microsoft IIS 5.0 and 6.0 allows remote attackers to poison the web cache, bypas |
  | CVE-2005-0871 | 2005-05-02 | 5.0 | MEDIUM | calendar_scheduler.php in Topic Calendar 1.0.1 module for phpBB, when running on |

### Microsoft ASP.NET 
- 🔎 NVD search: https://nvd.nist.gov/vuln/search/results?query=Microsoft%20ASP.NET
- 💥 Exploit-DB: https://www.exploit-db.com/search?q=Microsoft%20ASP.NET
- 📚 CVE/MITRE: https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=Microsoft%20ASP.NET

  **Recent CVEs (NVD keyword `Microsoft ASP.NET`, newest first):**

  | CVE | Published | CVSS | Severity | Summary |
  |-----|-----------|------|----------|---------|
  | CVE-2011-3414 | 2011-12-30 | 7.8 | HIGH | The CaseInsensitiveHashProvider.getHashCode function in the HashTable implementa |
  | CVE-2011-1253 | 2011-10-12 | 9.3 | HIGH | Microsoft .NET Framework 1.0 SP3, 1.1 SP1, 2.0 SP2, 3.5.1, and 4, and Silverligh |
  | CVE-2011-1977 | 2011-08-10 | 4.3 | MEDIUM | The ASP.NET Chart controls in Microsoft .NET Framework 4, and Chart Control for  |
  | CVE-2011-1978 | 2011-08-10 | 4.3 | MEDIUM | Microsoft .NET Framework 2.0 SP2, 3.5.1, and 4 does not properly validate the Sy |
  | CVE-2011-0664 | 2011-06-16 | 9.3 | HIGH | Microsoft .NET Framework 2.0 SP1 and SP2, 3.5 Gold and SP1, 3.5.1, and 4.0, and  |

### ASP.NET Framework 2.0.50727
- 🔎 NVD search: https://nvd.nist.gov/vuln/search/results?query=ASP.NET%20Framework%202.0.50727
- 💥 Exploit-DB: https://www.exploit-db.com/search?q=ASP.NET%20Framework%202.0.50727
- 📚 CVE/MITRE: https://cve.mitre.org/cgi-bin/cvekey.cgi?keyword=ASP.NET%20Framework%202.0.50727

  **Recent CVEs (NVD keyword `ASP.NET Framework (product-only fallback)`, newest first):**

  | CVE | Published | CVSS | Severity | Summary |
  |-----|-----------|------|----------|---------|
  | CVE-2021-43853 | 2021-12-22 | 8.7 | HIGH | Ajax.NET Professional (AjaxPro) is an AJAX framework available for Microsoft ASP |
  | CVE-2021-32831 | 2021-08-30 | 7.5 | HIGH | Total.js framework (npm package total.js) is a framework for Node.js platfrom wr |
  | CVE-2018-8356 | 2018-07-11 | 5.5 | MEDIUM | A security feature bypass vulnerability exists when Microsoft .NET Framework com |
  | CVE-2015-6099 | 2015-11-11 | 4.3 | MEDIUM | Cross-site scripting (XSS) vulnerability in ASP.NET in Microsoft .NET Framework  |
  | CVE-2015-2526 | 2015-09-09 | 5.0 | MEDIUM | Microsoft .NET Framework 4.5, 4.5.1, 4.5.2, and 4.6 allows remote attackers to c |

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