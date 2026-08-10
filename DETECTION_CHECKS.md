# Active Detection Checks (executed by the tool)
# فحوصات الكشف النشطة التي تنفّذها الأداة

هذه ٣٠ فحص كشف **غير تدميري** يشغّلها `ayed_smart_api_hunter.py` فعليًا في
**Phase 5 (Active Detection)** — كل فحص إمّا **طلب واحد** أو **تحليل محلي** لما نزّلناه،
بدون استغلال أو brute أو DoS. النتائج تظهر كـ leads في `report.md/json/html`
والـ`playbook.md`، وتربط بأرقام التقنيات في `RED_TEAM_METHODOLOGY.md` و
`RED_TEAM_50_ADVANCED_PLAYS.md`. للتحقق التفاعلي: `--verify`. للتعطيل: `--no-active-checks`.

## أ) كشف سلبي لكل رد (تحليل محلي، بلا طلبات إضافية) — P81–P96
| # | الفحص | ماذا يكشف |
|---|---|---|
| P81 | verbose_error | آثار stack trace / رسائل خطأ قواعد بيانات (ORA/SQLSTATE/Traceback) |
| P82 | tech_disclosure | كشف الخادم/التقنية (Server, X-Powered-By, X-AspNet-Version) |
| P83 | directory_listing | صفحة "Index of /" (autoindex) |
| P84 | internal_ip | عناوين RFC1918 داخلية في الرد/الهيدرات |
| P85 | pii_email | تسريب عناوين بريد (≥3) في الرد |
| P86 | clickjacking | غياب X-Frame-Options و CSP frame-ancestors |
| P87 | content_sniffing | غياب X-Content-Type-Options: nosniff |
| P88 | cacheable_json | JSON بلا no-store/private (مؤشر Web Cache Deception) |
| P89 | no_ratelimit_headers | لا هيدرات rate-limit على نقطة حسّاسة (auth/api) |
| P90 | secret_in_url | توكن/كلمة مرور/JWT في سلسلة الاستعلام |
| P91 | mixed_content | مورد http:// داخل صفحة https |
| P92 | cms_fingerprint | بصمة WordPress/Drupal/Laravel/Django/Rails/Next.js |
| P93 | weak_hsts | HSTS بدون includeSubDomains |
| P94 | cookie_samesite | كوكي بلا SameSite |
| P95 | debug_mode | APP_DEBUG/Whoops/Werkzeug/phpinfo في الرد |
| P96 | cors_wildcard_creds | OPTIONS يعلن ACAO:* مع credentials |

## ب) كشف على مستوى المضيف (طلبات مقيّدة) — P97–P104
| # | الفحص | ماذا يكشف |
|---|---|---|
| P97 | security_txt | وجود `/.well-known/security.txt` |
| P98 | oidc | كشف `/.well-known/openid-configuration` + نقاط OAuth |
| P99 | cors_null | قبول `Origin: null` |
| P100 | cors_naive_match | عكس Origin تعسّفي (subdomain/suffix trust) |
| P101 | trace_method | تفعيل TRACE (XST) |
| P102 | host_header_reflection | انعكاس `X-Forwarded-Host` في الجسم/Location |
| P103 | wp_user_enum | تعداد مستخدمين عبر `/wp-json/wp/v2/users` |
| P104 | admin_panels | وجود لوحات `/admin`,`/debug`,`/metrics`,`/status` |

## ج) كشف الملفات المكشوفة والأسرار — P105–P110
| # | الفحص | ماذا يكشف |
|---|---|---|
| P105 | source_maps | توفّر `*.js.map` (إعادة بناء المصدر) |
| P106 | backup_files | ملفات `.bak`/`~` للمسارات المكتشفة |
| P107 | exposed_files | `.git/config`, `.env`, `/actuator`, نسخ احتياطية |
| P108 | secrets | مفاتيح AWS/Google/Slack/GitHub/GitLab/Stripe/Twilio/SendGrid/Mailgun/Firebase/NPM + private keys في JS |
| P109 | jwt_weakness | تفكيك JWT: alg=none، HMAC confusion، jku/kid، لا exp، claims حسّاسة |
| P110 | takeover_fingerprint | بصمات subdomain takeout (S3/Heroku/GitHub Pages/...) |

## د) محرك الكشف المتقدم (data-driven) — 130+ فحص إضافي
يُنفَّذ ضمن نفس Phase 5، بيانات قابلة للتوسعة:

| المجموعة | العدد | أمثلة |
|---|---|---|
| **BODY_SIGNATURES** | 25 | أخطاء SQL لكل قاعدة (MySQL/Postgres/MSSQL/Oracle/SQLite)، stack traces (PHP/ASP.NET/Java/Python/Ruby/Node)، صفحات debug (phpinfo/Whoops/Werkzeug/Django/Rails)، تسريبات (مسارات، AWS ARN، S3، GCP SA، private keys، connection strings) |
| **HEADER_RULES** | 10 | X-Debug/X-Runtime/X-Powered-By/X-AspNet-Version، X-Cache (poisoning)، Via، cf-ray، debug cookies |
| **TECH_FINGERPRINTS** | 28 | WordPress/Drupal/Joomla/Magento/Shopify، Laravel/Django/Rails/Express/Next/Nuxt/Angular/React/Spring/ASP.NET، Nginx/Apache/IIS، Cloudflare/Akamai/Fastly/Imperva/F5/Sucuri/AWS-WAF/ModSecurity |
| **SENSITIVE_PATHS** | 45 | `.git/.svn/.hg`، `.env*`، `.aws/credentials`، `.npmrc/.htpasswd`، `settings.py/appsettings.json/web.config`، `docker-compose/Dockerfile`، `*.sql` dumps، `actuator/*` (env/heapdump/mappings)، `metrics/debug/vars`، swagger/openapi |
| **SECRET_PATTERNS** | 18 | AWS/Google/Slack/GitHub/GitLab/Stripe/Twilio/SendGrid/Mailgun/Firebase/NPM + private keys + generic assignments |

### محرك الفحص النشط (اختياري: `--intrusive`) — علامات حميدة فقط
| الصنف | كيف يكشف | تقنية |
|---|---|---|
| XSS_REFLECTION | علامة `ayedx…<b>` تنعكس بدون ترميز | T11 |
| SQLI_ERROR | إضافة `'` واحدة تُظهر خطأ قاعدة بيانات | T14 |
| SSTI | `{{7*7}}` يُقيَّم إلى 49 | T13 |
| LFI_TRAVERSAL | `../../etc/passwd` يطابق `root:x:0:0` (إثبات فقط) | T12 |

> `--intrusive` **مطفأ افتراضيًا**، يرسل قيمة فحص واحدة حميدة لكل بارامتر (بحد أقصى ٣ بارامترات × ١٢ رابط)، بدون brute أو حمولات تدميرية أو time-based. استعمله على أهداف مصرّح بها فقط.

**الإجمالي:** ‏٣٠ فحص (P81–P110) + ‏130+ فحص من المحرك المتقدم = **‏160+ فحص كشف تنفّذه الأداة فعلاً.**

## ملاحظة الحدود
كل ما سبق **كشف** لا **استغلال**. تقنيات الاستغلال الفعلي (request smuggling،
brute، race، SSRF لبيانات السحابة، تزوير JWT مُرسل، dependency confusion...) موثّقة
في ملفات المنهجية كخطوات **يدوية بموافقة مكتوبة**، ولا تُؤتمت هنا لأنها قد تضرّ الهدف
أو مستخدمين آخرين أو تخرج عن النطاق المصرّح.

## أمثلة تشغيل مُتحقّق منها (على مواقع اختبار عامة)
```bash
# فحص كامل + حفظ لمجلد + كل الصيغ
python3 ayed_smart_api_hunter.py https://httpbin.org --out ./out --formats md,html,json,playbook

# فحص + تحقق تفاعلي بعد التقرير (GET آمن فقط عند التأكيد)
python3 ayed_smart_api_hunter.py https://target --verify

# بدون مرحلة الكشف النشط
python3 ayed_smart_api_hunter.py https://target --no-active-checks
```
