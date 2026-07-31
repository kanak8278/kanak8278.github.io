---
title: "They'll Spend Anything — Sources & Calculations"
permalink: /blog/theyll-spend-anything-sources/
layout: single
author_profile: false
toc: true
toc_label: "Sections"
classes:
  - content-page
  - tex2jax_ignore
excerpt: "Every number behind the rant: source links, exchange rates, the arithmetic for each Fermi estimate, and the things I got wrong."
---

Companion to [**They'll Spend Anything**](/blog/theyll-spend-anything/). Every figure in that piece, where it came from, and how I got there.
{: .notice--info}

Everything behind *They'll Spend Anything*. Compiled 1 August 2026.

**Conventions used throughout**

- Where a source reports a USD figure, I use theirs. Where I converted from rupees myself, I used **₹88 = $1** (roughly the mid-2026 rate) and marked it *[converted]*.
- ₹1 crore = ₹10 million = ~$113,600 at ₹88/$. ₹1 lakh crore = ₹1 trillion = ~$11.36bn.
- **No PPP adjustment anywhere.** Where purchasing power actually matters (fab tooling), it's flagged in the text.
- "Pledge" means announced future commitment, not deployed capital. Flagged wherever used.
- Figures I calculated rather than found are marked **[my estimate]** with the arithmetic shown.

---

## 1. Semiconductors

| Claim | Value | Source |
|---|---|---|
| Dholera fab node | Mostly 90nm; Tata says 28–110nm range, "start with 55nm and 90nm" | [Business Standard, 17 Jul 2026](https://www.business-standard.com/amp/companies/news/tata-electronics-to-launch-india-s-first-chip-fab-using-older-technology-126071700161_1.html) |
| Dholera commercial production | Mid-2028 (slipped from end-2026) | Ashwini Vaishnaw, via Business Standard, Jul 2026 |
| Tata's stated starting node in its FY25 annual report | 28nm | Tata Sons annual report, y/e March 2025 |
| Dholera investment / capacity | ₹91,000 cr (~$11bn); 50,000 wafers/month; 20,000 jobs | [Tata Group](https://www.tata.com/newsroom/business/first-indian-fab-semiconductor-dholera) |
| Adani–Tower Semiconductor $10bn fab | **Strategic pause since Jan 2026**, subsidy terms unresolved | Industry reporting, Jan 2026 |
| Leading-edge node years | 90nm: 2004 (Intel Prescott, TSMC Nexsys). 28nm: 2011. 7nm: 2018. 3nm: Dec 2022 (TSMC N3). 2nm: Q4 2025 (TSMC N2) | TSMC press releases; [3nm process, Wikipedia](https://en.wikipedia.org/wiki/3_nm_process) |
| TSMC Arizona | $12bn (2020) → $65bn → $165bn (6 fabs) → $265bn with the 2025 addition | [TSMC Arizona](https://www.tsmc.com/static/abouttsmcaz/index.htm); [Blackridge](https://www.blackridgeresearch.com/project-profiles/tsmc-arizona-fab-united-states-us-details-cost-expansion-latest-update) |
| Samsung Taylor, Texas | $17bn (2020) → ~$25bn | [SiliconANGLE, Jul 2025](https://siliconangle.com/2025/07/04/tsmc-reportedly-accelerates-investment-arizona-chip-complex-samsung-delays-texas-fab/) |
| China Big Fund III | ¥344bn / **$47.5bn**, May 2024, 19 state investors, MoF 17% | [SCMP](https://www.scmp.com/tech/tech-war/article/3264612/tech-war-chinas-big-fund-iii-brings-us475-billion-fresh-outlay-nations-semiconductor-supply-chain); [Caixin](https://www.caixinglobal.com/2024-05-28/china-piles-475-billion-into-big-fund-iii-to-boost-chip-development-102200633.html) |

**"24 years" gap** = 2028 (Dholera commercial production, 90nm) − 2004 (industry 90nm volume) = 24.
**"15-year-old node"** = 2026 (original Dholera target) − 2011 (28nm volume) = 15.

**Sub-5nm fab in India: $25–40bn [my estimate].** Anchored on the two real precedents above. Samsung's single leading-edge fab landed at ~$25bn; TSMC's per-fab Arizona cost, backing out packaging and R&D, is in the same band and rose 13× from its original plan. First-in-market friction (workforce, permitting, water, power reliability, supply chain) adds cost. **Caveat that matters:** 70–80% of a fab's cost is tooling — ASML lithography, Applied Materials and Lam deposition/etch — priced in USD on a global market. India gets no domestic-cost discount on that portion. And ASML's EUV tools are export-controlled, so the top of the range may not be purchasable at any price.

---

## 2. R&D spending

| Claim | Value | Source |
|---|---|---|
| India R&D, % of GDP | 0.64% | [Economic Survey 2025-26, via BusinessToday, 29 Jan 2026](https://www.businesstoday.in/economic-survey/story/indias-rd-spend-at-06-of-gdp-due-to-low-contribution-from-private-sector-economic-survey-513453-2026-01-29) |
| Business share of R&D | India 41% · China 77% · US 75% · South Korea 79% | Economic Survey 2025-26 (same) |
| R&D % GDP, others | Korea 4.91% · US 3.48% · China 2.43% | Economic Survey 2025-26 (same) |
| **Conflicting newer figure** | FY24: 0.84% of GDP, ₹2.45 lakh cr total; private ₹1.27 lakh cr vs govt ₹1.18 lakh cr — private ahead for the first time | [Business Standard, 30 Jul 2026](https://www.business-standard.com/economy/news/r-d-spending-in-india-only-0-84-of-gdp-in-fy24-govt-tells-parliament-126073001491_1.html) |

**On the 0.64% vs 0.84% conflict:** these are different vintages and methodologies (Economic Survey tabled Jan 2026 vs a July 2026 parliamentary answer covering FY24). I used the Economic Survey figures in the charts because the country comparison comes from the same table, and flagged the higher number in the text and chart footnote. The newer number is genuinely better news and I don't want to hide it. It does not change the conclusion: 0.84% is still roughly a third of China's intensity.

### Corporate R&D intensity

| Company | R&D | Revenue | Intensity | Source |
|---|---|---|---|---|
| Huawei | CNY 179.7bn / **$24.76bn** (2024) | ~$118–119bn | **20.8%** (company-stated) | [Huawei 2024 Annual Report](https://www.huawei.com/en/annual-report/2024) |
| Samsung Electronics | KRW 37.74tn / **$26.2bn** (2024) | ~$232bn | **11.3%** (company-stated) | [Seoul Economic Daily](https://en.sedaily.com/finance/2026/02/14/samsung-electronics-posts-record-26b-rd-spending-in-2024) |
| Reliance Industries | **₹4,185 cr** (FY25, audited) ≈ **$476M** *[converted]* | ₹5,57,163 cr standalone ≈ $63.3bn | **0.75%** **[my calculation]** | RIL Integrated Annual Report 2024-25 |

**The RIL source, exactly.** [RIL Integrated Annual Report 2024-25](https://www.ril.com/reports/RIL-Integrated-Annual-Report-2024-25.pdf), Board's Report, Annexure on Conservation of Energy / Technology Absorption / Foreign Exchange (the statutory disclosure required by Rule 8(3) of the Companies (Accounts) Rules, 2014), section **"(iv) Expenditure incurred on Research and Development"**:

| | ₹ crore |
|---|---|
| a) Capital | 2,652 |
| b) Revenue | 1,533 |
| **Total** | **4,185** |

Standalone turnover, same report (Standalone Statement of Profit and Loss): Value of Sales & Services **₹5,57,163 cr**; Revenue from Operations ₹5,32,792 cr. Consolidated Value of Sales & Services was ₹10,71,174 cr.

**Arithmetic:**
- 4,185 / 5,57,163 = **0.751%** → the headline figure
- 1,533 / 5,57,163 = **0.275%** → recurring/expensed R&D only
- Ratio to Huawei: 20.8 / 0.751 = **27.7×**, stated as "twenty-eight times"

**Three caveats, all of which cut in different directions.**

1. *This is the standalone entity.* The Companies Act disclosure covers RIL the company, not the group. R&D done inside Jio Platforms and Reliance Retail is excluded and is not separately disclosed anywhere I could find. Group R&D intensity is therefore **somewhat higher than 0.75%** — I don't know by how much, and nobody outside Reliance does.
2. *Two-thirds of it is capital spending.* ₹2,652 cr of the ₹4,185 cr is buildings and equipment. Huawei's and Samsung's percentages are expensed R&D off the income statement. The strictly like-for-like comparison is Reliance's ₹1,533 cr = **0.27%**, which is 76× below Huawei. I lead with the 0.75% figure because it is the one most generous to Reliance, and it is still damning.
3. *Business mix is a real objection.* Huawei is a pure technology company. Reliance's largest segments are refining and retail, which are structurally less R&D-intensive everywhere in the world. Fair. It is also the point: choosing which business to be in is the decision under discussion.

**Correction from my own first pass.** I originally published 0.34%, computed from a third-party FY24 figure (~₹3,600 cr) divided by *consolidated* FY25 revenue. That mixed bases — a standalone-scope R&D number over a group-scope denominator — and understated Reliance by roughly 2.2×. The audited standalone-over-standalone figure is 0.75%.

---

## 3. AI: models, money, training costs

| Claim | Value | Source |
|---|---|---|
| DeepSeek R1 RL run | **$294,000**, 512× H800 | DeepSeek, peer-reviewed in *Nature*, Sept 2025; [CNN](https://www.cnn.com/2025/09/19/business/deepseek-ai-training-cost-china-intl) |
| DeepSeek V3 pre-training | **$5.58M**, 2,048× H800, ~2 months, 2.79M GPU-hours | Same; [The Register's caveat](https://www.theregister.com/2025/09/19/deepseek_cost_train/) |
| High-Flyer's GPU purchase | ¥1bn (~**$147M**), 10,000 A100s, 2021 | [Liang Wenfeng, Wikipedia](https://en.wikipedia.org/wiki/Liang_Wenfeng); ChinaTalk |
| DeepSeek's first outside round | ¥51bn (~$7.5bn) at ~¥400bn valuation | 2026 reporting |
| GPT-4 training compute | ~$78M | Stanford HAI AI Index 2025 / Epoch AI |
| Llama 3.1 405B | ~$170M | Same |
| Gemini Ultra | ~$191M | Same |
| Alibaba AI/cloud capex | ¥380bn (~**$53bn**) over 3 years | [Alibaba Cloud](https://www.alibabacloud.com/blog/alibaba-to-invest-rmb380-billion-in-ai-and-cloud-infrastructure-over-next-three-years_602007) |
| Qwen open models | 400+ released; >1bn HF downloads; 200,000+ derivatives; passed Llama as default open family | [SCMP](https://www.scmp.com/tech/big-tech/article/3339765/alibaba-reaffirms-open-source-ai-commitment-tech-giant-hails-qwen-achievements) |
| Liang Wenfeng net worth, Jul 2026 | **$36.0bn** (+$19.9bn YTD, +120%); richest AI-model founder globally; 8th-richest in China | [TechNode, 15 Jul 2026](https://technode.com/2026/07/15/deepseek-founder-liang-wenfengs-fortune-rises-to-36-billion/), citing Bloomberg Billionaires Index |
| Driver of that rise | DeepSeek's $7.4bn round, Jun 2026, at a $50bn valuation (up from $10bn in April); Liang holds ~78% | Same |
| Liang, early 2026 | $4.6bn (Hurun) — reflects that list's earlier valuation cut-off, not a fall in wealth | Same |
| India's share of global AI funding | **0.6%** | [Business Standard, Feb 2026](https://www.business-standard.com/markets/news/india-s-share-of-global-funding-pie-in-ai-stands-at-0-6-shows-data-126021901360_1.html) |
| Indian AI startup funding, 2025 | $1.34bn across 198 deals | Tracxn |
| Chinese AI startup PE/VC, 2025 | $6.7bn (56% of Asia); excludes state programmes | Crunchbase / regional reporting |

**The most important caveat in this whole document.** The $5.58M figure is **compute for one pre-training run.** It excludes the GPUs (that's the separate $147M line), salaries, data, infrastructure and every failed run before it. Quoting it as "the cost of building a frontier model" is misleading and *The Register* was right to say so. The honest version, and the one the piece uses: **~$147M of hardware plus single-digit millions per training run** — which is still one to two orders of magnitude below the "billions of dollars" figure the Indian policy conversation is built on.

### Open-weight model scale (July 2026)

| Model | Params | Origin |
|---|---|---|
| Kimi K3 (Moonshot) | 2,800B | China |
| GLM-5.2 (Zhipu, MIT licence) | 753B | China |
| Sarvam-105B | 105B | India |
| BharatGen Param2 (17B MoE, 22 languages) | 17B | India |

**The comparison used is Kimi K3 vs Sarvam-105B: 2,800 / 105 = 26.7×, stated as 27×.** An earlier draft compared Kimi against BharatGen's 17B (165×), which was cherry-picking — measuring China's largest against India's *smallest*. Sarvam-105B is the largest model India has, so it's the fair comparator, and 27× is the honest number.

Parameter count is a crude capability proxy and the chart footnote says so — a well-trained small model can beat a large one. The gap illustrated is one of ambition and resourcing, not benchmark performance.

### Funding Sarvam instead of building from scratch

$1bn into Sarvam = **0.55%** of Ambani + Adani's combined $183bn, and **3.3×** Sarvam's entire $300m Series B. HCLTech's actual $150m cheque is what took Sarvam to unicorn status in June 2026, which is the precedent the argument rests on.

---

## 2b. Elon Musk, as the risk-appetite comparison

| Claim | Value | Source |
|---|---|---|
| PayPal proceeds, 2002 | **$180m** from eBay's $1.5bn acquisition | Musk's own account, repeated consistently |
| How he allocated it | **$100m SpaceX · $70m Tesla · $10m SolarCity** | Same |
| Late 2008 | Tesla down to **$9m** in the bank, days from insolvency; SpaceX had failed **three** consecutive Falcon 1 launches; Musk borrowing from friends for personal rent | [CBS News](https://www.cbsnews.com/news/billionaire-elon-musk-on-2008-the-worst-year-of-my-life/); Fox Business; Startup Archive |
| What saved it | Fourth Falcon 1 reached orbit; NASA awarded a **$1.6bn** Commercial Resupply Services contract | Same |
| Net worth, mid-July 2026 | **~$833bn** (Bloomberg Billionaires Index), +$214bn YTD; more than $500bn clear of second place | [Bloomberg Billionaires](https://www.bloomberg.com/billionaires/profiles/elon-r-musk/) |
| Trillionaire milestone | Briefly passed **$1 trillion** on the SpaceX IPO, 12 June 2026, which valued SpaceX at **$1.77 trillion** — the largest listing in history; fell back as the stock corrected | Bloomberg; IPO reporting |

**Why he's in the piece.** As an illustration of risk appetite, not an endorsement. His net worth is volatile and any figure here is a snapshot. The structural point is the ordering: the fortune followed the bet rather than funding it — the same ordering as Liang Wenfeng, and the opposite of the Indian pattern the piece describes.

---

## 3b. Indian advertising spend

| Item | Value | Source |
|---|---|---|
| India total ad market, 2025 | **₹1,55,105 cr** (~$17.6bn); digital ₹93,156 cr (60%). *Cut from the piece — see note below.* | [Pitch Madison Advertising Report 2026](https://www.medianews4u.com/indias-ad-market-crosses-%E2%82%B91-55-lakh-crore-in-2025-digital-now-60-of-adex-madison-report-2026/) |
| Hindustan Unilever, FY25 A&P | **₹6,028 cr** (~$685M), down from ₹6,380 cr | Storyboard18 |
| Maruti Suzuki, FY25 ad + sales promotion | **₹1,742.6 cr** (~$198M), +11.42% | Storyboard18 |
| ITC, FY25 ad + promotion | **₹1,331.69 cr** (~$151M), −3.89% YoY | Storyboard18 |
| Ads placed inside quick-commerce apps, 2025 | **₹4,000 cr** (~$455M), up from ₹1,325 cr — a 202% jump | Pitch Madison 2026 |
| Tata IPL title sponsorship | ₹2,500 cr for 2024–28 = **₹500 cr/yr** (~$57M/yr) | Wikipedia / SportsPro |
| My11Circle, IPL associate sponsor | **₹125 cr/yr** (~$14.2M) — the league's most valuable central deal after Tata | SportsPro |
| IPL 2025 total ad revenue | ~₹4,500 cr (~$511M) | Industry reporting |
| Nvidia one-day market-cap loss, 27 Jan 2025 | **$589bn** — largest single-day loss in market history; stock −17% | [CNBC](https://www.cnbc.com/2025/01/27/nvidia-sheds-almost-600-billion-in-market-cap-biggest-drop-ever.html); [Bloomberg](https://www.bloomberg.com/news/articles/2025-01-27/asml-sinks-as-china-ai-startup-triggers-panic-in-tech-stocks) |

**"Everything DeepSeek spent" = $152.9M:** $147M (10,000 A100s, 2021) + $5.58M (V3 pre-training) + $0.294M (R1 RL). Against ITC's $151.3M — within 1%.

**"Runs" as a unit.** Throughout the advertising chart I express budgets as multiples of DeepSeek's $5.58M V3 pre-training run. **This is a unit of measurement, not an economic claim.** Ad budgets do not convert into research labs; an FMCG company's advertising is how it generates the revenue that pays for everything else, and cutting it to zero would not produce a frontier model. The comparison is there to make the *scale* legible, because "$5.58 million" is a number most readers cannot place.

**A comparison I cut.** An earlier draft ran India's *entire* national advertising market (₹1,55,105 cr / $17.6bn) against the training run: 3,159 runs a year. It's arithmetically correct and rhetorically useless — a whole-economy figure spanning every company and sector isn't comparable to one firm's project budget, and stretching it that far invites the reader to distrust the company-level comparisons that actually do hold. Company against company is the honest version, so that's what the piece keeps.

**Reliance's own advertising line is not in this table** because RIL does not disclose it separately — it sits inside "other expenses" in the consolidated accounts. I looked.

### Indian AI programmes

| Item | Value | Source |
|---|---|---|
| Sarvam Series B | $234M first close of a $300M round, $1.5bn post-money, 15 Jun 2026; HCLTech $150M for >10% | [HCLTech](https://www.hcltech.com/press-releases/sarvam-raises-234-million-first-close-300-million-series-b-15-billion-valuation); [TechCrunch](https://techcrunch.com/2026/06/15/sarvam-becomes-indias-newest-ai-unicorn-with-234-million-funding-round-led-by-hcltech/) |
| Sarvam-105B training | 1,000+ H100s at Yotta's Shakti cluster; IndiaAI allocated 4,000+ H100s; 12T tokens, 22 languages | [Forbes, 7 Mar 2026](https://www.forbes.com/sites/janakirammsv/2026/03/07/india-can-train-a-sovereign-model-but-still-cannot-prove-it-works/) |
| IndiaAI Mission | ₹10,372 cr (~$1.18bn *[converted]*; commonly reported as $1.25bn); 38,000+ GPUs empanelled | [PIB](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2245069&reg=3&lang=1); [IndiaAI](https://indiaai.gov.in/news/cabinet-approves-india-ai-mission-at-an-outlay-of-rs-10-372-crore) |
| BharatGen | ₹988.6 cr (~$112M *[converted]*), largest single IndiaAI allocation | [Inc42](https://inc42.com/features/bharatgen-ceo-on-why-models-alone-cant-help-india-gain-in-the-ai-race/) |
| Krutrim | ₹2,000 cr (~$227M *[converted]*) committed Feb 2025, ₹10,000 cr more promised | [Business Standard](https://www.business-standard.com/companies/news/ola-founder-bhavish-aggarwal-to-invest-rs-2-000-crore-in-ai-firm-krutrim-125020400838_1.html) |
| **Krutrim pivot** | Paused chip design and foundation-model work, pivoted to AI cloud | [MediaNama, May 2026](https://www.medianama.com/2026/05/223-krutrim-ai-cloud-chip-ai-model-work/) |
| Smallest.ai | $13M Series A (31 Jul 2026), >$21M total; competes with ElevenLabs/Cartesia | [TechCrunch](https://techcrunch.com/2026/07/31/smallest-ai-raises-13m-to-build-ultra-fast-voice-ai-that-sounds-genuinely-human/) |
| Bolna | $6.3M seed, Jan 2026 | TechCrunch, Jan 2026 |

**Nilekani quotes.** "Let the big boys in Silicon Valley do it, spending billions of dollars. We will use it to create synthetic data, build small language models quickly" — Meta Build with AI Summit, Bengaluru, Oct 2024, on a panel with Yann LeCun ([Forbes India](https://www.forbesindia.com/article/leadership/while-other-people-build-all-the-llms-we-will-make-sure-it-works-for-people-nandan-nilekani/94493/1)). "Foundation models are not the best use of your money" — BusinessToday, Dec 2024. Public disagreement from Manish Gupta (Google Research India) and Aravind Srinivas (Perplexity), both via Outlook Business, Jan 2025.

---

### Stanford Global AI Vibrancy (the "India is third" line)

| Country | Score |
|---|---|
| United States | 78.60 |
| China | 36.95 |
| **India** | **21.59** |

2024 index, published in the 2025 edition of Stanford HAI's [Global AI Vibrancy Tool](https://www.visualcapitalist.com/cp/ai-competitiveness-by-country/). India rose from 7th (2023) to 3rd, overtaking the UK, South Korea, Singapore and Japan. India's score is **27.5% of the US** and **58% of China's**.

The ranking is not fake and I don't argue that it is. It aggregates research output, talent, investment and governance — categories where India genuinely is strong. The objection in the piece is narrower: a composite index weighted toward *talent and paper counts* will rank India highly while telling you nothing about whether the country has built a frontier artefact, and it is quoted as though it does.

---

## 4. Data-centre pledges

| Item | Value | Source |
|---|---|---|
| Reliance | ₹10 trillion / **$110bn** over 7 years; Jamnagar, 120MW live H2 2026 | [TechCrunch, 19 Feb 2026](https://techcrunch.com/2026/02/19/reliance-unveils-110b-ai-investment-plan-as-india-ramps-up-tech-ambitions/); [Forbes](https://www.forbes.com/sites/yessarrosendar/2026/02/19/billionaire-mukesh-ambani-steps-up-ai-push-with-110-billion-data-center-investment-plan/) |
| Adani | ~**$100bn** | TechCrunch, Feb 2026 |
| Government target | $200bn+ over two years | Ashwini Vaishnaw, via WION, Jun 2026 |
| Microsoft / Google India AI infra | $17.5bn / $15bn | WION, Jun 2026 |
| Chip dependency | Jamnagar on Nvidia Blackwell; IndiaAI's 38,000+ GPUs almost all H100/H200 | TechCrunch; WION, Jun 2026 |

Chart 07 totals: hosting = 110,000 + 100,000 + 17,500 + 15,000 = **$242,500M**. Building = 300 (Sarvam) + 227 (Krutrim) + 112 (BharatGen) = **$639M**. Ratio ≈ **380:1**.

**These are pledges.** Announced February 2026, spanning seven years. Not deployed capital. Re-check in 2028.

---

## 5. Space

| Item | Value | Source |
|---|---|---|
| DoS memo | 14 Jul 2026 — stop accepting resignations/VRS from scientists on Gaganyaan and strategic missions; all cases referred to DoS. Reverses a 2020 delegation to centre directors | [WION](https://www.wionews.com/india-news/isro-scientist-resignations-gaganyaan-talent-exodus-government-memo-1784127615955); [BW People](https://www.bwpeople.in/article/isro-scientists-on-strategic-missions-face-tougher-resignation-process-615499) |
| Departures | 100–120 in recent months; ~80 from UR Rao Satellite Centre | Same |
| All Indian spacetech funding, ever | **$871M** across 241 rounds, 285 companies (to July 2026) | [Business Standard / Tracxn, 28 Jul 2026](https://www.business-standard.com/companies/start-ups/india-s-spacetech-startups-raise-871-million-as-private-sector-takes-off-126072801130_1.html) |
| Top-funded | Skyroot $150M (unicorn, May 2026) · Pixxel $96M · AgniKul $76M · Digantara $67M | Same |
| **Vikram-1 orbital launch** | **18 July 2026**, Sriharikota — first privately built Indian rocket to reach orbit (~450 km). Made India the third country with private orbital launch capability, after the US and China. Mission "Aagaman"; 350 kg LEO class | [Space.com](https://www.space.com/space-exploration/launches-spacecraft/skyroot-aerospace-india-first-private-orbital-launch-vikram-1); SpaceDaily |
| Skyroot funding and investors | **$150M** total; Sherpalo Ventures, GIC (Singapore sovereign), Temasek (Singapore state holding), BlackRock. Unicorn after $50M Series C, May 2026 | Tracxn, 2026 |
| Indian researcher return rate | **Only 26.6%** of internationally mobile Indian researchers return. Barriers cited: procurement rules, multi-layered file approvals, tendering delays, rigid hierarchy, limited autonomy | [Scientific mobility patterns of Indian researchers, arXiv:2509.18069](https://arxiv.org/pdf/2509.18069); 360info; Policy Circle |
| Mangalyaan | **$74M**, Mars orbit first attempt; NASA MAVEN $582–671M | [Business Standard](https://www.business-standard.com/india-news/what-makes-india-s-space-missions-cost-less-than-hollywood-sci-fi-movies-124110400430_1.html) |
| Chandrayaan-3 | **$75M** | Same |

---

## 6. Cricket, weddings, wealth

| Item | Value | Source |
|---|---|---|
| IPL digital rights 2023–27 | ₹23,758 cr to Viacom18 (Reliance); ~**$2.9–3.0bn** as reported at 2022 rates. Total IPL media rights ₹48,390 cr | [ESPNcricinfo](https://www.espncricinfo.com/story/disney-star-and-viacom-18-share-the-spoils-in-6-billion-dollar-plus-ipl-rights-deal-1319863) |
| Mumbai Indians enterprise value | **$2.2bn** (Houlihan Lokey IPL Valuation Study 2025). Standalone *brand* value $242M — a different metric | [Houlihan Lokey](https://hl.com/insights/ipl-valuation-study-brand-valuation-of-ipl-and-franchisees/) |
| IPL overall | Business value $18.5bn (+12.9%); league brand value $3.9bn | Same |
| Indiawin Sports FY24 | Revenue ₹737 cr; net profit ₹109 cr | [CricTracker, from filings](https://www.crictracker.com/cricket-news/mumbai-indians-make-inr-109-crore-profit-in-fy-2024/) |
| Anant Ambani wedding | Estimates **$150M–$600M** (some report up to $1bn). Includes $150M cruise, $6M Rihanna, $10M Bieber | [Wikipedia](https://en.wikipedia.org/wiki/Wedding_of_Anant_Ambani_and_Radhika_Merchant); NBC News |
| Vantara | ₹1,200–2,500 cr capital (**$150–300M**); ₹150–200 cr/yr to run; 3,000 acres | India.com; Suryatara |
| Hike / Rush | $261M raised (Tiger Global, Tencent, Bharti); shut down Sept 2025 after the RMG ban; ~$4M left | [TechCrunch](https://techcrunch.com/2025/09/13/hike-once-a-unicorn-shuts-down-as-india-cracks-down-on-real-money-gaming/); Entrackr |
| Net worth, Apr 2026 | Adani **$92.6bn** · Ambani **$90.8bn** (Bloomberg Billionaires Index). Forbes' March 2026 list put Ambani at $99.7bn | Sunday Guardian / Bloomberg |
| Adani's 2021 wealth gain | **+$49bn in one year** — more than Musk, Bezos and Arnault added combined | Hurun Global Rich List 2022 |
| Hurun Rich List 2026 (India) | 308 billionaires, 3rd globally. Ambani ₹9.8 lakh cr (**$111bn**) · Adani ₹7.5 lakh cr (**$85bn**) · Roshni Nadar Malhotra ₹3.2 lakh cr (**$36.4bn**) | [Business Standard, Mar 2026](https://www.business-standard.com/india-news/hurun-rich-list-2026-india-now-has-308-billionaires-ranks-third-globally-126030700305_1.html) |

**The "Liang would be India's third-richest" claim — read the caveat.** Indian figures are Hurun (March 2026, rupees, converted at ₹88/$). Liang's $36.0bn is Bloomberg (July 2026). **That is a different index on a different date**, and the two methodologies value private holdings differently. At $36.0bn against Roshni Nadar Malhotra's $36.4bn he is effectively tied for third, not clearly ahead of her — the piece says "level with," which is the accurate framing. If you want a single-index comparison, Hurun's own early-2026 number for Liang ($4.6bn) predates the DeepSeek repricing and isn't comparable either. The defensible version of the claim: **on any current reckoning Liang sits in the same bracket as India's third-richest person, having started from nowhere in 2021.**
| Zepto | FY25 loss ₹3,367 cr (~$383M); burned ~$500M in under a year | Reuters / StartupTalky |

**MI running cost [my calculation]:** ₹737 cr revenue − ₹109 cr net profit = **₹628 cr** of costs = **~$71M** *[converted]*. This is total operating cost, not just player salaries, and it's the Indian franchise only — MI Cape Town, MI Emirates and MI New York were all loss-making in FY24 and are excluded.

**Combined net worth used in the piece:** 92.6 + 90.8 = **$183.4bn**, rounded to $183bn.

**Cross-checks:**
- 2,900 / 871 = **3.33×** ("Reliance paid 3× India's entire space industry to stream cricket")
- 600 / 871 = **68.9%** ("about 70% of every rupee India's private rocket industry has raised") — this uses the *upper* wedding estimate; at the low end ($150M) it's 17%
- **"Another fortnight and you'd have bought the whole sector" is deliberately approximate.** The gap between the wedding's top estimate and total private rocketry funding is $871M − $600M = **$271M**. The wedding's events ran roughly March–July 2024, so "a fortnight" is a rhetorical rounding of what another $271M of the same spending would take, not a costed figure. The defensible claim is the ratio: **~70%**, at the upper estimate. At the lower estimate the line does not hold and I'd drop it.
- 910 / 5.58 = **163** ("165 DeepSeek-V3 training runs" — rounded)
- 2,900 / 5.58 = **520**

---

## 6b. The next generation

| Heir | Role / spend | Source |
|---|---|---|
| Akash Ambani | Chairman, Reliance Jio Infocomm; MD, Jio Platforms (>$100bn) | Forbes India; The Federal |
| Isha Ambani | Exec Director, Reliance Retail (~$30bn revenue) + RCPL. **₹1 lakh cr revenue target by FY30**; **₹30,000 cr (~$3.41bn) into food parks over 3 years** for beverages, chocolates, biscuits, staples. RCPL revenue doubled to ₹22,000 cr in FY26; Campa now India's 4th-largest CSD brand | [Business Standard, 19 Jun 2026](https://www.business-standard.com/companies/news/reliance-consumer-products-targets-rs-1-lakh-crore-revenue-by-fy30-isha-ambani-126061900998_1.html) |
| Anant Ambani | Reliance New Energy; Vantara (3,000 acres, $150–300M, ₹150–200 cr/yr) | India.com; Suryatara |
| Karan Adani | MD, Adani Ports & SEZ | [Adani Group](https://www.adani.com/about-us/leadership/karan-adani) |
| Jeet Adani | Director, Adani Airports; also leads AI, data centres, defence, petrochemicals, copper | [Adani Group](https://www.adani.com/about-us/leadership/jeet-adani) |
| Aryaman Vikram Birla | **Chairman, RCB** — Aditya Birla-led consortium (with Times Group, Bolt Ventures, Blackstone BXPE) acquired 100% of the IPL and WPL franchises from United Spirits, **March 2026, ₹16,663 cr / $1.78bn** — the most expensive franchise sale in IPL history | [ESPNcricinfo](https://www.espncricinfo.com/story/rcb-sold-for-usd-1-78-billion-to-aditya-birla-times-of-india-led-consortium-1529082); [Aditya Birla Group](https://www.adityabirla.com/media/press-releases/aditya-birla-group-the-times-of-india-group-bolt-ventures-and-blackstone-to-acquire-cricket-franchise-royal-challengers-bengaluru/) |
| Ananya Birla | Svatantra Microfin; music; beauty and lifestyle brands | Zee News; NewsX |
| Parth Jindal | JSW Cement, JSW Paints; 50% Delhi Capitals; Bengaluru FC | NewsX |
| Kavin Bharti Mittal | **Hike Messenger** (2012) — WhatsApp rival, 100M users, $1.4bn valuation after a $175M Tencent/Foxconn Series D in Aug 2016; messenger shut 2021; became **Rush**, a real-money gaming platform (~10M users, ~$500M gross revenue over four years); **entire company shut Sept 2025** after the Promotion and Regulation of Online Gaming Act banned RMG. $261M raised, ~$4M left | [Hike Messenger, Wikipedia](https://en.wikipedia.org/wiki/Hike_Messenger); [TechCrunch](https://techcrunch.com/2025/09/13/hike-once-a-unicorn-shuts-down-as-india-cracks-down-on-real-money-gaming/); Entrackr |
| Roshni Nadar Malhotra | HCLTech; $150M into Sarvam AI | HCLTech press release, Jun 2026 |

**Key ratios:**
- RCB $1,780M ÷ all Indian spacetech $871M = **2.04×** ("more than twice everything India's private space industry has raised")
- Reliance food parks $3,410M vs the $3,000M three-year lab estimate = the food parks are **1.14× larger**
- ₹30,000 cr ÷ 88 = **$3,409M**, stated as $3.4bn

**A fairness note on Jeet Adani.** He does formally hold the AI and data-centre brief, so it's not true that no heir touches AI at all. The piece says so. The objection is that the brief is infrastructure — data centres running foreign silicon — not model or chip development, which is the same landlord distinction drawn earlier.

---

## 7. Campa Cola / beverages

| Item | Value | Source |
|---|---|---|
| Campa brand acquisition | ₹22 cr, 2022 | Wikipedia |
| RCPL beverage capex | **₹6,000–8,000 cr** by March 2027; 10–12 new plants | [Business Standard, 19 Jun 2025](https://www.business-standard.com/companies/news/reliance-consumer-products-beverage-expansion-coca-cola-pepsi-campa-125061900145_1.html) |
| RCPL FY25 revenue | ₹11,500 cr; Campa and Independence each crossed ₹1,000 cr | Same |

**Correction from the earlier draft.** The ₹8,000 cr is capex for **RCPL's beverage business overall** — Campa plus Independence and other brands — not for Campa alone, and it's an upper bound on a ₹6,000–8,000 cr range. The earlier draft treated it as a Campa-specific commitment. ₹8,000 cr = **~$910M** *[converted]*.

---

## 8. Universities

| Item | Value | Source |
|---|---|---|
| Harvard freeze | $2.2bn grants + $60M contracts (Apr 2025); litigated at ~$2.6–2.7bn; a further ~$1bn was floated | [PBS](https://www.pbs.org/newshour/politics/judge-reverses-trump-administrations-cuts-of-billions-in-research-funding-to-harvard); [Harvard Chan School](https://hsph.harvard.edu/news/trump-administration-freezes-2-2-billion-in-grants-to-harvard/) |
| Outcome | Federal judge reversed the freeze Sept 2025; a majority of funds restored by Oct 2025; administration appealed Apr 2026 | Harvard Crimson |
| All 23 IITs, FY26 | ₹11,349 cr (**~$1.29bn** *[converted]*), up from ₹10,324.5 cr in FY25 | [Shiksha, union budget 2026](https://www.shiksha.com/news/engineering-union-budget-2026-govt-allocates-83-562-cr-for-schools-55-727-cr-for-higher-ed-iit-funding-rises-iisc-iiit-see-cuts-blogId-220818) |
| IISc, FY26 | ₹845 cr (**~$96M**) — **cut** from ₹900 cr | Same |
| IISERs, FY26 | ~₹1,540 cr less a ₹137 cr cut ≈ ₹1,400 cr; **~$175M** used, conservative | Careers360 / Shiksha; **lowest-confidence figure in this document** |

**Important honesty note.** The Harvard money was **frozen, litigated, and largely restored** — it was not a permanent cut. The comparison in the piece is to the *size of one political fight* versus India's entire annual elite-science allocation, and the chart footnote says so. Anyone quoting it as "the US cut more than India spends" is overstating it.

Total: 1,290 + 175 + 96 = **$1.56bn**, stated as "about $1.6 billion."

**PPP flips this comparison, and the piece now says so.** Converted at market rates (₹88/$) the Indian total is $1.56bn against Harvard's $2.6bn. Converted at World Bank purchasing-power rates (roughly ₹23/$), the same ₹13,734 crore is worth about **$5.97bn** of local buying power — more than double the Harvard figure. So the "bigger than India's entire elite science system" line is true in **nominal dollars only**, and the post carries a footnote saying exactly that.

The nominal figure still carries information, because the internationally-priced share of a research budget gets no PPP discount: instruments, GPUs, cleanroom tooling, journal subscriptions, and researcher salaries, which compete against Zurich and Seattle rather than local rates. But anyone arguing India's elite institutions are better resourced in real terms than this section suggests is correct, and I'd rather concede it than defend it.

---

## 9. Korea, 1983

| Claim | Source |
|---|---|
| Tokyo Declaration, 8 Feb 1983; Lee Byung-chul aged 73, terminally ill; ~$400M of group reserves into semiconductors; opposed by his senior team; Intel called it delusional; Mitsubishi published five reasons Samsung would fail | [Korea JoongAng Daily](https://www.koreajoongangdaily.com/opinion/lee-byungchulls-final-gamble-at-73/12780062); [Lee Byung-chul, Wikipedia](https://en.wikipedia.org/wiki/Lee_Byung-chul) |
| First 1Mb DRAM 1986; 4Mb 1988; 16Mb 1990; 64Mb 1992 (~6 months ahead of Toshiba/NEC/Hitachi); passed Toshiba as largest DRAM maker 1992 | [Samsung Electronics, Wikipedia](https://en.wikipedia.org/wiki/Samsung_Electronics) |
| Lee died Nov 1987, 4 years 10 months after the declaration | Wikipedia |
| Korea's 1953 per-capita income below Somalia's and Haiti's; Samsung + SK Hynix now make ~2/3 of world memory | SpaceDaily |

---

## 10. The $3bn lab **[my estimate — the least defensible number in the piece]**

| Line | Amount | Basis |
|---|---|---|
| 500 researchers @ $1.2M blended | $600M/yr | Meta research-scientist total-comp bands, **$305K (IC4) – $581K (IC6)**, per [Levels.fyi](https://www.levels.fyi/companies/meta/salaries/research-scientist) (last updated 30 Jul 2026), plus OpenAI's reported ~$1.5M average stock comp per employee. Deliberately **not** the $100M+ superstar packages. Includes a diaspora-return premium. |
| ~10,000 H100-class GPUs | ~$300M | ~$30K/unit. Cross-checked against High-Flyer's actual $147M for 10,000 A100s in 2021. |
| Data, power, ops, tooling | ~$200M | Roughly 3× Sarvam's stated $300M three-year budget, annualised. |
| **Year 1** | **~$1.1bn** | |
| **Three years** | **~$3bn** | Assumes flat run-rate; a real lab's compute line would grow. |

**Where this could be wrong:** the compute line could be 2× off in either direction depending on buy-vs-rent, generation, and whether you're training or also serving. The talent line assumes you can actually recruit 500 such people to India, which is an assumption about desirability, not money — arguably the real constraint and one this number cannot capture. Treat $3bn as an order-of-magnitude claim: **single-digit billions, not tens of billions.**

**Ratios:** 3,000 / 110,000 = **2.73%** of Reliance's announced data-centre pledge. 3,000 / 111,000 = **2.7%** of Mukesh Ambani's net worth on Hurun 2026 (3.3% on Bloomberg's April 2026 figure of $90.8bn — the piece says "roughly 3%", which holds on either index). 3,000 / 871 = **3.4×** all Indian spacetech funding ever.

**A comparison I got wrong and corrected.** An earlier draft said the $3bn lab was "less than what Zepto burned delivering groceries in twelve months ($500M/year)." That is backwards — $3bn is **6× larger** than $500M. The corrected version compares like periods: Zepto burning ~$500M/year for three years is **$1.5bn**, so the three-year lab is **2× Zepto's three-year burn**, not less than it.

---

## 11. Corrections made from the first draft

Things the earlier version got wrong or overstated, fixed here:

1. **Private R&D share was 36%.** It's **41%** (Economic Survey 2025-26), and a July 2026 parliamentary answer puts India's overall R&D at 0.84% of GDP with private spend now exceeding government spend. Both changes cut against the argument; both are in the piece.
2. **"Adani has committed $10bn with Tower Semiconductor"** — presented as live. It has been in **strategic pause since January 2026**.
3. **Tata's Dholera fab described as mature-node generally.** The specific and much worse fact: it's opening at **90nm**, two nodes below what Tata told shareholders, and has slipped to **mid-2028**.
4. **₹8,000 cr described as Campa's budget.** It's RCPL's whole beverage capex, and it's the top of a ₹6,000–8,000 cr range.
5. **Hike's funding was "$250M+."** It's **$261M**.
6. **MI's $2.2bn** needed disambiguating — it's *enterprise value*, not brand value ($242M). Both are in the Houlihan Lokey study.
7. **Harvard framed as a cut.** It was frozen, then reversed in court and largely restored.
8. **DeepSeek's $5.58M was not in the first draft at all**, and it's the single most important number in the argument — it's what makes "foundation models cost billions" false.
9. **Krutrim's shutdown was missing.** The best-documented case of an Indian founder actually attempting chips and models, and quitting, was absent.
10. **xAI section removed.** The first draft leaned on Musk as the model. The Samsung 1983 story is better: it's an Asian family conglomerate, in a poorer country, and it worked. It also removes the need to relitigate Musk.

---

## 12. What I could not verify

- **Reliance's exact FY25/FY26 R&D line.** RIL's integrated annual report is a large PDF I could not extract cleanly. The ₹3,600 cr FY24 figure is third-party. Anyone with the report should check page-level R&D disclosure and correct the intensity chart.
- **Per-franchise IPL enterprise values** beyond MI. Houlihan Lokey publishes brand values by franchise but business values mostly in aggregate.
- **Whether the 0.64% or 0.84% R&D figure is methodologically correct.** Both are government sources, seven months apart.
- **The true Vantara capital cost.** Published estimates range ₹1,200–2,500 cr with no audited figure. The chart uses the midpoint.
- **Total private conglomerate participation in Indian deep-tech rounds.** I checked the top space and AI startups' disclosed investor lists and found no Indian conglomerate leading a round other than HCLTech in Sarvam. That's an absence of evidence from public cap tables, not a proof of absence.
