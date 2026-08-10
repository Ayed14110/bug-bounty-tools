# 50 Advanced Red-Team Attacker Plays
# ٥٠ فكرة هجومية متقدمة (عقلية مهاجم)

> **Authorized scope only / للنطاق المُصرّح به فقط.** هذه أفكار عميقة يُنفّذها
> المُختبِر ضمن برنامج bug bounty أو أصول يملكها، **تحقّقًا يدويًا غير تدميري**،
> بحسابات وموارد يملكها. لا DoS، لا تدمير، لا مسّ ببيانات مستخدمين حقيقيين.
> هذه تكمّل `RED_TEAM_METHODOLOGY.md` (التقنيات 1–30). الترقيم هنا P31–P80.

الصيغة: **الهدف (كمهاجم) ← كيف تنفّذها ← الأثر**.

---

## أولًا: GraphQL و APIs العميقة

**P31 — GraphQL Alias-based Brute.** استخدم aliases في استعلام واحد لتنفيذ عشرات
محاولات المصادقة/التخمين وتجاوز rate-limit. راقب أي alias نجح. الأثر: تجاوز الحدّ،
حشو بيانات اعتماد.

**P32 — GraphQL Field-level Authz Bypass.** اطلب حقولًا حساسة عبر fragments أو
inline على نوع مشترك حيث ينسى الـresolver فحص التفويض حقلًا حقلًا. الأثر: تسريب
حقول (بريد، هاتف، توكن).

**P33 — GraphQL Query Complexity (recon).** قِس عمق/تكلفة الاستعلامات المتداخلة
لتحديد غياب حدّ التعقيد (احتمال DoS) — **رصد فقط، بلا إغراق**. الأثر: مؤشر DoS.

**P34 — GraphQL Mutation IDOR.** طبّق منطق IDOR على mutations (`updateUser(id:...)`)
لا على queries فقط. الأثر: تعديل كائنات الغير.

**P35 — Cross-endpoint UUID Leakage.** اجمع UUIDs المسرّبة في ردّ endpoint واستعملها
كمعرّفات "سرّية" في endpoint آخر يفترض عدم قابلية التخمين. الأثر: IDOR على GUID.

**P36 — Second-order IDOR.** معرّف تُخزّنه في خطوة (ملف، عنوان) يُستهلك لاحقًا في
سياق آخر بلا إعادة فحص ملكية. الأثر: وصول غير مصرّح مؤجّل.

**P37 — Tenant Isolation Bypass (SaaS).** في نظام متعدد المستأجرين بدّل
`org_id/workspace_id/tenant` مع إبقاء جلستك. الأثر: اختراق عزل المستأجرين.

**P38 — Blind IDOR via Diff.** حين لا تُرجع البيانات مباشرة، ميّز الملكية عبر فرق
الحجم/التوقيت/رمز الحالة (200 vs 404 vs 403). الأثر: أوراكل وجود/ملكية.

**P39 — Numeric ID → Identity Oracle.** endpoint يحوّل `id` إلى اسم/بريد يكشف
تعداد المستخدمين. الأثر: تعداد هوية، تجهيز تصيّد.

**P40 — API Type Juggling.** أرسل نوعًا غير متوقّع (`"1"` vs `1`، مصفوفة بدل نص،
`null`، `true`) لتجاوز فحوص أو كسر منطق. الأثر: تجاوز تحقّق/تفويض.

---

## ثانيًا: مصادقة متقدمة

**P41 — JWT jku/x5u SSRF.** وجّه رأس `jku`/`x5u` إلى JWKS تتحكم به لتوقيع توكنك.
الأثر: انتحال هوية كاملة.

**P42 — JWT kid Injection.** استغل `kid` عبر path traversal أو SQLi لاختيار مفتاح
معروف/فارغ. الأثر: تزوير توقيع.

**P43 — OAuth redirect_uri Path Tricks.** جرّب `//attacker`, `/redirect?url=`,
مطابقة prefix ضعيفة، parameter pollution على `redirect_uri`. الأثر: سرقة code →
استيلاء حساب.

**P44 — OAuth Referrer Token Leak.** التوكن/الـcode في الـURL يتسرّب عبر `Referer`
لموارد طرف ثالث في الصفحة. الأثر: تسريب توكن.

**P45 — SAML Signature Wrapping (XSW).** أعد لفّ عناصر الـAssertion بحيث يوقّع
المدقّق جزءًا ويقرأ التطبيق جزءًا آخر. الأثر: انتحال هوية.

**P46 — SAML Assertion Replay.** أعد إرسال Assertion صالحة (غياب فحص التكرار/الوقت).
الأثر: إعادة استخدام جلسة.

**P47 — Cookie Prefix Bypass.** اختبر احترام `__Host-`/`__Secure-` وإمكان تثبيت
كوكي من subdomain. الأثر: تثبيت جلسة، تجاوز نطاق.

**P48 — Account Linking / Login CSRF.** اربط هوية IdP يتحكم بها المهاجم بحساب
الضحية (غياب `state`). الأثر: استيلاء حساب عبر الربط.

**P49 — Reset Token Race.** استهلك رابط الاستعادة عبر طلبات متزامنة قبل الإبطال.
الأثر: تجاوز استخدام لمرة واحدة.

**P50 — OTP/Magic-link Entropy.** حلّل عشوائية/طول/توقيت الرموز على حسابك لتقدير
قابلية التخمين — **بلا brute فعلي**. الأثر: مؤشر تخمين OTP.

---

## ثالثًا: SSRF وحقن متقدم

**P51 — Blind SSRF via DNS Rebinding.** اسم يحلّ أولًا لعنوانك ثم لعنوان داخلي بين
الفحص والجلب (TOCTOU على DNS). الأثر: تجاوز allow-list، وصول داخلي.

**P52 — SSRF Filter Bypass.** تجاوز الفلاتر بترميزات IP (عشري/ثماني/`0x`, `[::]`,
`127.0.0.1.nip.io`, enclosed-alphanumerics). الأثر: وصول للشبكة الداخلية.

**P53 — Cloud Metadata (IMDS).** بعد إثبات SSRF ضمن نطاق مسموح، اقرأ IMDS
(بما فيها تدفّق IMDSv2 بالتوكن). الأثر: بيانات اعتماد سحابية.

**P54 — CRLF → Response Splitting.** حقن `%0d%0a` في بارامتر ينعكس في رأس
الاستجابة (Set-Cookie/Location). الأثر: تسميم رؤوس، XSS، تثبيت جلسة.

**P55 — Prototype Pollution.** لوّث `__proto__`/`constructor.prototype` عبر JSON
أو query لتغيير سلوك التطبيق (server أو client). الأثر: تجاوز تحقّق، أحيانًا RCE/XSS.

**P56 — Injection through GraphQL Resolver.** حمولة تمرّ من متغيّر GraphQL إلى
استعلام backend (SQL/NoSQL). الأثر: حقن غير مباشر.

**P57 — CSV / Formula Injection.** في التصدير، قيمة تبدأ بـ`=/+/-/@` تُنفَّذ في
Excel عند الضحية. الأثر: تنفيذ عند فتح الملف.

**P58 — XSLT / XPath Injection.** حقن في تحويلات XML/استعلامات XPath. الأثر: قراءة
ملفات، تجاوز مصادقة XPath.

**P59 — Header/Parameter Cloaking.** رؤوس/بارامترات مكرّرة أو غير قياسية
(`Content-Length` مزدوج، مسافات) تُفسَّر مختلفًا بين WAF والتطبيق. الأثر: تجاوز WAF.

**P60 — Unicode/Normalization Bypass.** استغل توحيد Unicode/case-folding
(`ﬃ`→`ffi`، dotless i) لتجاوز فلاتر أو مطابقة بريد/مستخدم. الأثر: تجاوز فلاتر/تعارض هوية.

---

## رابعًا: طبقة HTTP والبروكسي

**P61 — HTTP/2 Smuggling (H2.CL / H2.TE).** استغل ترجمة HTTP/2→1.1 في الحافة
لتهريب طلب. الأثر: تسميم طلبات، تجاوز أمن.

**P62 — Client-side Desync.** تهريب مدفوع بالمتصفح بلا تحكّم في الـfront-end. الأثر:
اختطاف طلبات مستخدمين.

**P63 — Cache Key Confusion.** استغل تطبيع مفتاح الكاش (بارامترات/مسار/رؤوس غير
مُفتاحة) لتخزين محتوى مسموم. الأثر: Cache Poisoning موجّه.

**P64 — Web Cache Deception on JSON APIs.** `/api/me/nonexistent.css` يجعل رد
JSON حسّاسًا يُخزّن كملف ثابت عام. الأثر: تسريب بيانات موثّقة.

**P65 — Range / Partial Content Abuse.** رؤوس `Range` غير متوقعة تكشف تسريبات
ذاكرة/تجاوز قيود. الأثر: تسريب، تجاوز.

**P66 — Cross-Site WebSocket Hijacking.** WS بلا فحص Origin/توكن CSRF. الأثر: قراءة/
إرسال رسائل باسم الضحية.

**P67 — Reverse-Proxy Path Confusion.** استغل فرق تفسير المسار بين البروكسي
(nginx alias، traefik، `..;/`) والتطبيق للوصول لمسارات محمية. الأثر: تجاوز ACL.

**P68 — Virtual Host / SNI Routing.** اطلب أسماء مضيف داخلية عبر نفس الـIP
(`Host: internal-app`) أو SNI للوصول لتطبيقات غير منشورة. الأثر: كشف تطبيقات داخلية.

**P69 — Method / Version Ambiguity.** `HTTP/0.9`، أفعال غير قياسية، `TRACE`،
`X-HTTP-Method-Override` تتجاوز الضوابط. الأثر: تجاوز تحكّم وصول.

**P70 — Second-request Poisoning.** استغل keep-alive/connection reuse بحيث يؤثر
طلبك في الطلب التالي على نفس الاتصال. الأثر: تسميم استجابة.

---

## خامسًا: السحابة والعميل الحديث

**P71 — Bucket Enumeration & ACL.** عدّد S3/GCS/Azure buckets من أسماء الشركة
وافحص List/Read/Write ACL. الأثر: تسريب/تعديل ملفات.

**P72 — Exposed CI/CD & VCS.** ابحث عن `.git/`, `.github/workflows`, توكنات
GitLab/Jenkins, artifacts. الأثر: تسريب مصدر/أسرار، سلسلة توريد.

**P73 — Kubernetes/Kubelet Exposure (recon).** منافذ API/kubelet/dashboard
مكشوفة بلا مصادقة. الأثر: سيطرة على عناقيد.

**P74 — Serverless Event Injection.** حقن في مصدر حدث Lambda/Function (S3 event,
SNS, API GW) يتجاوز التحقّق. الأثر: تنفيذ منطق غير مقصود.

**P75 — Firebase / NoSQL Open Rules.** قواعد Firebase/Firestore عامة تسمح
قراءة/كتابة. الأثر: تسريب/تعديل بيانات.

**P76 — postMessage Origin Bypass.** مستمع `message` بلا فحص `origin` يقبل بيانات
من إطار المهاجم. الأثر: XSS/سرقة توكن client-side.

**P77 — DOM Clobbering & Script Gadgets.** استغل عناصر DOM/gadgets في مكتبات
(sanitizer bypass) لتنفيذ سكربت رغم CSP. الأثر: XSS متقدم.

**P78 — Dependency Confusion.** اسم حزمة داخلي منشور عامًّا برقم أعلى يُسحب في
البناء. الأثر: RCE في سلسلة التوريد (بلاغ فقط، بلا نشر خبيث فعلي).

**P79 — Source Map & Secret Reuse.** أعد بناء الشيفرة من `.js.map`، واختبر إعادة
استخدام مفتاح API عبر endpoints/بيئات مختلفة. الأثر: كشف منطق/توسيع صلاحيات.

**P80 — Mobile/Thick-client API Keys.** استخرج مفاتيح/توكنات مضمّنة في تطبيق
الجوال أو ملفات التهيئة، واختبر صلاحيتها على الـAPI العام. الأثر: وصول موثّق مسرّب.

---

## قواعد الاشتباك
- ✅ ضمن النطاق المكتوب فقط · حسابات/موارد تملكها · أقل أثر يثبت الثغرة.
- ⛔ لا DoS/إغراق · لا تدمير أو تعديل بيانات مستخدمين حقيقيين · لا نشر حزم/محتوى خبيث.
- 📝 وثّق كل خطوة (curl/HAR/collaborator logs) لتقرير قابل لإعادة الإنتاج، وأبلغ فورًا
  عن أي بيانات حسّاسة تظهر عرضًا دون تخزينها.

## كيف تربطها بالأداة
شغّل `ayed_smart_api_hunter.py` أولًا لبناء سطح الهجوم والأعلام، ثم اختر الطبقة الأعلى
(CRITICAL/HIGH) وطبّق الأفكار المناسبة من هذا الملف يدويًا، مستعينًا بجدول الربط في
`RED_TEAM_METHODOLOGY.md`.
