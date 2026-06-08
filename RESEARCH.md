# RESEARCH.md — Voice AI Platform: Product Research & Opportunity Map

> Last updated: 2026-06-08. Based on live market research, competitor analysis, and our own build/test data.

---

## What We Built (Core Engine)

Not a booking agent — a **configurable voice AI engine** that handles real phone calls with tool-calling, state machines, and per-tenant config. The booking bot is just the first template.

| Capability | Status |
|------------|--------|
| Inbound calls via Twilio | ✅ |
| Multilingual auto-detect (Hindi/English) | ✅ |
| LLM with tool-calling (GPT-4o) | ✅ |
| State machine flows (Pipecat Flows) | ✅ |
| Per-tenant config (persona, services, hours) | ✅ |
| Budget + guardrail enforcement | ✅ |
| Caller memory (returning callers) | ✅ |
| Devanagari Hindi + script-based detection | ✅ |
| Cost: **$0.06/call (~₹5)** | ✅ |

### Live-Tested Performance (2026-06-08)

| Metric | Value |
|--------|-------|
| Avg LLM TTFB (GPT-4o) | ~0.78s |
| Avg TTS TTFB (Sarvam) | ~0.55s |
| Total perceived turn latency | ~1.5–2.5s |
| Full Hindi booking call duration | ~70–100s |
| Per-call cost | $0.063 (₹5.4) |
| Monthly cost at 500 calls | ~$33 (₹2,800) |

---

## Market Size

| Market | Size | Source |
|--------|------|--------|
| Global Voice AI | $22.5B (2026) | [Voice AI Wrapper](https://voiceaiwrapper.com/insights/voice-ai-market-analysis-trends-growth-opportunities) |
| India Voice AI | $153M (2024) → **$957M by 2030** (35.7% CAGR) | [NextMSC](https://www.nextmsc.com/report/india-voice-assistant-market-3375) |
| India MSMEs | **63 million** businesses, 113M employees | [IBEF](https://www.ibef.org/industry/msme) |
| Missed calls | SMBs miss **62%** of calls; 85% never call back | [411 Locals](https://411locals.us/small-business-owners-dont-answer-62-of-phone-calls/) |
| Cost of missed calls | **$126K/year** average lost revenue per SMB | [Phone2](https://www.phone2.io/post/true-cost-of-missed-calls) |
| ROI breakeven | **3.2 months** median for voice AI deployment | [Ringly](https://www.ringly.io/blog/voice-ai-statistics-2026) |
| US SMB adoption | 34% deployed/piloting AI voice (Q1 2026, up from 8% in 2024) | [Ringly](https://www.ringly.io/blog/voice-ai-statistics-2026) |

---

## Competitor Pricing (We're 5–10x Cheaper)

| Platform | Per-minute cost | Model | Notes |
|----------|----------------|-------|-------|
| **Aircall** | $0.49–$0.99/min | Seat + minutes | Enterprise, most expensive |
| **JustCall** | $0.83–$0.99/min | Seat + minute bundles | CRM-integrated |
| **Synthflow** | $0.15–$0.24/min | Component-based | No-code, good for non-technical |
| **Bland AI** | $0.11–$0.14/min | All-in per-minute | Cheapest US platform |
| **Retell AI** | $0.13–$0.31/min | Component-based | Best turn-taking quality |
| **Vapi** | $0.13–$0.31/min | Platform fee + components | Most developer-flexible |
| **Bolna** | $0.05–$0.07/min | Subscription bundles | India-based, closest competitor |
| **Cartesia** | $0.06/min + STT/TTS | Component-based | New entrant, free LLM during early access |
| **Us** | **~$0.037/min** | Own stack | Full stack, multilingual, India-first |

Sources: [Zeeg Pricing Guide](https://zeeg.me/en/blog/post/ai-voice-agent-pricing-guide), [Pickaxe Comparison](https://pickaxe.co/post/top-ai-voice-agents), [Retell Blog](https://www.retellai.com/blog/best-voice-ai-providers)

### Per-Call Cost Breakdown (Our Stack)

| Service | Cost/call | % of total |
|---------|-----------|------------|
| GPT-4o LLM | $0.044 | 70% |
| Sarvam STT | $0.012 | 19% |
| Twilio media | $0.007 | 11% |
| Sarvam TTS | $0.0004 | <1% |
| **Total** | **$0.063** | |

### LLM Cost Comparison (Per Call)

| LLM | Cost/call | Notes |
|-----|-----------|-------|
| Claude Opus | $0.270 | Overkill for voice |
| GPT-4o | $0.044 | Current — good quality + speed |
| Claude Haiku 4.5 | $0.014 | Cheapest quality option |
| GPT-4o-mini | $0.003 | Cheapest overall, quality TBD |

---

## India-Specific Competitors

| Player | Strength | Weakness | Pricing |
|--------|----------|----------|---------|
| **Sarvam AI** | Best Indian language models (22 langs), we use their STT/TTS | Not a turnkey agent — models/APIs only | API pricing (STT ₹0.60/min, TTS ₹0.04/1K chars) |
| **Gnani AI** | Enterprise voice biometrics, BFSI | Expensive, enterprise-only | Custom enterprise pricing |
| **Caller Digital** | Turnkey voice agents for India | Newer, limited languages | Not publicly listed |
| **Bolna** | India-based, affordable | No multilingual auto-detection | $0.05–$0.07/min |
| **Haptik/Yellow.ai** | Large enterprise, WhatsApp bots | Voice is secondary to chat | Enterprise pricing |
| **Ozonetel** | Contact center platform, Indian languages | Legacy IVR-heavy, not AI-native | Per-seat licensing |

Sources: [Top Voice AI India 2026](https://caller.digital/blog/top-10-voice-ai-agents-india-2026), [Sarvam AI](https://www.sarvam.ai/products/conversational-agents)

### Market Gap

**No one owns multilingual voice AI for India's 63M SMBs.**
- Enterprise players (Gnani, Haptik) → too expensive for SMBs
- Model providers (Sarvam) → not building the turnkey product
- US platforms (Vapi, Retell) → don't support Indian languages natively
- Budget players (Bolna) → no multilingual auto-detection

We sit in the gap: **turnkey, multilingual, ₹5/call, India-first.**

---

## Product Directions

### Direction 1: Multi-Vertical Inbound Agent

Same engine, different flow config templates per industry. Each vertical is a config change, not a rebuild.

| Vertical | TAM in India | Why voice AI fits | Effort |
|----------|-------------|-------------------|--------|
| **Salons/Spas** | 2M+ salons | Appointment-heavy, no receptionist | ✅ Built |
| **Clinics/Doctors** | 1.5M+ clinics | 1 receptionist, 100+ daily calls | Config template |
| **Restaurants** | 7.5M+ restaurants | Peak-hour overflow, takeaway orders | Config template |
| **Real Estate** | 100K+ brokers | Miss 40%+ leads, each worth ₹5–50K | Config + lead capture flow |
| **Auto Workshops** | 5M+ workshops | No receptionist, scheduling-heavy | Config template |
| **Hotels/Homestays** | 300K+ properties | Multilingual guests, reservations | Config template |
| **Coaching/Tutoring** | 400K+ centers | Parent inquiries, trial class scheduling | Config template |
| **Legal/CA firms** | 500K+ practices | High-value consultation booking | Config template |
| **Gyms/Fitness** | 50K+ gyms | Class booking, membership inquiries | Config template |
| **Government** | Pan-India | Scheme info, passport/Aadhaar appointments, 22 languages | Custom flows |

### Direction 2: Outbound Calling Engine

Flip the direction — instead of answering calls, **make** calls. Same pipeline in reverse.

| Use Case | How it works | Monetization |
|----------|-------------|-------------|
| **Appointment reminders** | Call tomorrow's bookings, confirm/reschedule | ₹2/reminder — reduces 30% no-shows |
| **Payment/EMI reminders** | Soft collection for NBFCs, microfinance | ₹5–10/call or % of recovered amount |
| **Lead qualification** | Call website/ad leads, qualify, book demos | ₹50–100/qualified lead |
| **Feedback/NPS** | Post-visit satisfaction calls | ₹3/survey completed |
| **Re-engagement** | "It's been 30 days since your last visit" | ₹2/call |
| **Order confirmation** | E-commerce order/address verification | ₹1/call |
| **Campaign blasts** | New service launches, seasonal offers | Per-call + per-conversion |

Sources: [Aircall Use Cases](https://aircall.io/blog/ai-voice-agent-use-cases/), [AI Voice Agent Playbook](https://medium.com/@tuguidragos/the-ai-voice-agent-playbook-10-use-cases-beyond-cold-calling-e0a591694cbd)

### Direction 3: Debt Collection (High Value)

Massive market with clear AI fit:
- Agent churn rate: **31%** (all-time high in 2026)
- AI achieves **45–50% containment rates** in mature deployments
- India's NBFC/microfinance sector has millions of collection calls daily
- Regulatory compliance (RBI guidelines) can be enforced by guardrails

Key capabilities needed: payment negotiation flows, compliance guardrails, DND/time-of-day enforcement, multi-language soft collection.

Sources: [AI Debt Collection 2026](https://blog.getdarwin.ai/en/ai-voice-agents-debt-collections-2026), [Kompato Forecast](https://kompatoai.com/future-of-ai-in-debt-collection/)

### Direction 4: Insurance & Healthcare (High Value)

| Use Case | Impact | Data |
|----------|--------|------|
| **FNOL intake** (First Notice of Loss) | Automate claim intake over phone | 70% reduction in processing time |
| **Policy renewal calls** | Outbound reminders before expiry | Reduces lapse rate |
| **Hospital appointment scheduling** | Multi-department, multilingual | 70–80% Tier 1 automation |
| **Prescription refill requests** | Patient calls, AI processes | Saves nurse time |
| **Lab report follow-up** | "Your reports are ready" | Outbound automation |
| **Customer satisfaction** | 37% improvement in CSAT scores | Industry benchmark |

Insurance companies pay **$0.49–$0.99/min** for current solutions. We can offer the same at **₹3–5/min** — 10x cheaper.

Sources: [AI Insurance Voice 2026](https://www.brilo.ai/resources/best-ai-voice-agents-for-insurance), [AI Insurance Claims](https://www.ema.ai/additional-blogs/addition-blogs/top-ai-voice-agents-insurance-companies)

### Direction 5: Platform / API Play (Highest Ceiling)

Build a platform others build on:

| Component | What | Revenue model |
|-----------|------|---------------|
| **Flow builder** | No-code drag-and-drop conversation design | SaaS subscription |
| **Template marketplace** | "Salon template", "Clinic template" | Revenue share |
| **Integration marketplace** | Google Calendar, Zoho, Razorpay, WhatsApp | Connector fees |
| **Developer API** | Embed voice AI in any app | Per-minute API pricing |
| **White-label** | Agencies resell under their brand | License fee + per-minute |
| **Analytics dashboard** | Call insights, peak hours, sentiment trends | Premium add-on |

### Direction 6: WhatsApp + Voice Combo (India Moat)

India's #1 app is WhatsApp (500M+ users). Combine voice + WhatsApp:

- **Voice call** → booking confirmed → **WhatsApp confirmation** sent automatically
- **WhatsApp message** → "Call us to book" → voice agent handles it
- **Post-call summary** → sent via WhatsApp with booking details
- **Day-before reminder** → WhatsApp message with confirm/reschedule option
- **Missed call** → WhatsApp follow-up: "Sorry we missed your call, how can we help?"

No US competitor does this. WhatsApp Business API + our voice engine = uniquely Indian product.

### Direction 7: Regional Language Expansion

We have Hindi + English. Sarvam supports 22 Indian languages. Each new language opens a new geography with near-zero competition:

| Language | Speakers | Key Markets |
|----------|----------|-------------|
| Tamil | 80M | Tamil Nadu, Sri Lanka |
| Telugu | 83M | Andhra Pradesh, Telangana |
| Marathi | 83M | Maharashtra |
| Bengali | 230M | West Bengal, Bangladesh |
| Kannada | 44M | Karnataka |
| Gujarati | 55M | Gujarat |
| Malayalam | 38M | Kerala |
| Punjabi | 33M | Punjab |

Each language is a config change (add TTS voice + update language list + add greeting). The detection engine already handles script-based switching.

### Direction 8: Advanced Features (Differentiation)

| Feature | Description | Competitive edge |
|---------|-------------|-----------------|
| **Caller memory** | Remember returning callers, preferences, past bookings | "Welcome back, Priya! Last time you had a haircut with Rahul" |
| **Sentiment routing** | Detect frustration/anger → transfer to human | Reduces churn, builds trust |
| **Smart upselling** | "Since you're coming for a haircut, we have 20% off color this week" | Revenue uplift for business |
| **Voice biometrics** | Authenticate caller by voice (no OTP) | Security without friction |
| **Real-time translation** | Caller speaks Tamil, response in Tamil — one flow config | One flow, N languages |
| **Call analytics** | Peak hours, common requests, satisfaction trends | Data becomes the product |
| **Multi-agent handoff** | AI handles booking, transfers to human for complaints | Hybrid human + AI |
| **Proactive scheduling** | AI calls when regular appointment is due | Predictive, not reactive |
| **IVR replacement** | Replace "Press 1 for Hindi" trees with natural conversation | Every call center needs this |

---

## Monetization Strategies

### Pricing Models

| Model | Price point | Target | When |
|-------|------------|--------|------|
| **Freemium** | 50 calls/mo free | Land & expand, build trust | Day 1 |
| **Per-call** | ₹3–5/call | SMBs, pay-as-you-go | Day 1 |
| **Monthly plans** | ₹999 / ₹2,999 / ₹4,999 | Growing businesses | Month 3 |
| **Per-qualified-lead** | ₹50–100/lead | Real estate, insurance | Month 6 |
| **Outcome-based** | % of booking value or recovered debt | Performance alignment | Month 6 |
| **White-label** | ₹25K–1L/mo | IT companies, agencies | Month 9 |
| **Enterprise** | Custom | Hospital chains, insurers, banks | Year 1 |

### Industry Trend: Outcome-Based Pricing

The industry is moving from per-seat/per-minute to **outcome-based** models:
- **Intercom** charges $0.99/resolved ticket → hit 9-figure revenue
- **Salesforce Agentforce** hit $800M ARR with outcome-aligned pricing

We could charge **per-confirmed-booking** (₹10–20) instead of per-minute. At ₹5 cost and ₹15 charge, that's 3x margin with aligned incentives.

Sources: [Bessemer AI Pricing Playbook](https://www.bvp.com/atlas/the-ai-pricing-and-monetization-playbook), [SaaS Mag](https://www.saasmag.com/how-saas-companies-monetizing-ai-agents/), [McKinsey AI SaaS](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/upgrading-software-business-models-to-thrive-in-the-ai-era)

---

## Competitive Moat

| Advantage | Why it's hard to copy |
|-----------|----------------------|
| **₹5/call pricing** | Structural — own stack, Indian infra (Sarvam) pricing |
| **Multilingual auto-detect** | Built into the engine, not bolted on |
| **India-first** | US competitors won't localize for India anytime soon |
| **Sarvam integration** | 22 Indian languages ready to enable |
| **Open architecture (Pipecat)** | Swap any component (STT/LLM/TTS) as better/cheaper options appear |
| **WhatsApp combo** | No US competitor has this |
| **Outcome-based pricing** | Can afford it at ₹5/call cost basis |
| **Vertical templates** | Each new vertical is config, not code |

---

## Recommended Roadmap

### Phase 1: Multi-Vertical + Reminders (Now → 3 months)
- Ship 3–4 vertical templates (clinic, restaurant, real estate, auto workshop)
- Add outbound reminder calls (appointment confirmation, reduce no-shows)
- WhatsApp confirmation after booking
- Freemium (50 calls free) + per-call pricing (₹3–5)
- Analytics: basic call logs, booking conversion rate

### Phase 2: Outbound Engine + Languages (3–6 months)
- Full outbound calling: lead qualification, payment reminders, campaigns
- Add Tamil + Telugu (open TN, AP, Maharashtra markets)
- Analytics dashboard (peak hours, sentiment, common requests)
- Monthly subscription plans
- CRM integrations (Zoho, Google Calendar)

### Phase 3: Platform + High-Value Verticals (6–12 months)
- Insurance FNOL, debt collection flows
- Developer API, white-label program
- No-code flow builder
- Outcome-based pricing (per-booking, per-qualified-lead)
- Voice biometrics, sentiment routing

### Phase 4: Scale (12+ months)
- All 22 Indian languages
- Template marketplace
- Enterprise sales (hospital chains, insurers, banks)
- International expansion (SEA, Middle East — similar multilingual needs)
- Data/insights product (industry benchmarks from aggregated call data)

---

## Key Insight

> **No one owns multilingual voice AI for India's 63M SMBs.** Enterprise players are too expensive. Model providers aren't building turnkey products. US platforms don't speak Indian languages. The gap is wide open — and our ₹5/call cost basis makes outcome-based pricing viable where competitors can't afford to.

---

## Sources

- [Voice AI Market Size 2026](https://voiceaiwrapper.com/insights/voice-ai-market-analysis-trends-growth-opportunities)
- [India Voice Assistant Market](https://www.nextmsc.com/report/india-voice-assistant-market-3375)
- [India MSME Stats](https://www.ibef.org/industry/msme)
- [SMBs Miss 62% of Calls](https://411locals.us/small-business-owners-dont-answer-62-of-phone-calls/)
- [Cost of Missed Calls](https://www.phone2.io/post/true-cost-of-missed-calls)
- [Voice AI Statistics 2026](https://www.ringly.io/blog/voice-ai-statistics-2026)
- [AI Voice Agent Pricing Guide](https://zeeg.me/en/blog/post/ai-voice-agent-pricing-guide)
- [Top AI Voice Agents 2026](https://pickaxe.co/post/top-ai-voice-agents)
- [Voice AI Platforms India 2026](https://caller.digital/blog/top-10-voice-ai-agents-india-2026)
- [Sarvam AI](https://www.sarvam.ai/products/conversational-agents)
- [Aircall Use Cases](https://aircall.io/blog/ai-voice-agent-use-cases/)
- [AI Voice Agent Playbook](https://medium.com/@tuguidragos/the-ai-voice-agent-playbook-10-use-cases-beyond-cold-calling-e0a591694cbd)
- [AI Debt Collection 2026](https://blog.getdarwin.ai/en/ai-voice-agents-debt-collections-2026)
- [AI Insurance Voice Agents](https://www.brilo.ai/resources/best-ai-voice-agents-for-insurance)
- [Bessemer AI Pricing Playbook](https://www.bvp.com/atlas/the-ai-pricing-and-monetization-playbook)
- [SaaS AI Monetization](https://www.saasmag.com/how-saas-companies-monetizing-ai-agents/)
- [McKinsey AI SaaS Era](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/upgrading-software-business-models-to-thrive-in-the-ai-era)
- [Retell AI Blog](https://www.retellai.com/blog/best-voice-ai-providers)
