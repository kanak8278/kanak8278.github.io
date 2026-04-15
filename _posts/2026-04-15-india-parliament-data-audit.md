---
title: "The Disappearing Parliament: What India's Legislature Actually Does With Its Time"
date: 2026-04-15
categories:
  - blog
tags:
  - india
  - parliament
  - democracy
  - data
  - visualization
  - policy
layout: blog-post
toc: false
read_time: true
excerpt: "India's parliament sat for 55 days a year in the 17th Lok Sabha — the lowest of any full-term parliament. 35% of bills passed with under an hour of Lok Sabha debate. In Monsoon 2025, 14 bills became law; the longest LS debate was 34 minutes. Every number here is from a primary source. None are approximate."
---

<div style="border-left:3px solid #f59e0b; padding:12px 18px; background:rgba(245,158,11,0.06); border-radius:0 6px 6px 0; margin-bottom:2rem;">
<p style="margin:0; color:#d1b07a; font-size:0.96em; line-height:1.75;">
The 17th Lok Sabha (2019–2024) averaged <strong style="color:#f59e0b">55 sitting days a year</strong> — the lowest of any full-term parliament. The National Commission to Review the Constitution recommended a minimum of 120 days. India's Constitution recommends no minimum at all. Every number in this post is sourced directly from PRS Legislative Research, PIB press releases, or official parliamentary records. Where sources disagree, both figures are shown.
</p>
</div>

## The Same Parliament, Two Sessions Apart

2025 opened with a Budget Session that exceeded its scheduled time. Then the Monsoon Session functioned at 29% of its scheduled hours. Same parliament. Four months apart.

{% include ip-sessions.html %}

*PRS and PIB use different productivity methodologies — both are shown. The gap between the two sessions, by either measure, is not marginal.*

## The Bills That Took Minutes

In the Monsoon Session 2025, Parliament passed 14 bills (PRS count, excluding finance/appropriation). The chart below shows how much time Lok Sabha actually spent debating each of them, against the time formally allocated on the agenda.

{% include ip-bill-times.html %}

*The Income-Tax (No.2) Bill 2025 replaced the original Income Tax Bill (which had been referred to a Select Committee in the Budget Session). The replacement was introduced and passed in Lok Sabha in a single day, with 4 minutes of LS debate. The Merchant Shipping Bill waited 244 days after introduction, then received 20 minutes. Rajya Sabha, in contrast, gave several of these bills 1–2 hours each — which the Lok Sabha number obscures.*

## The 70-Year Decline

Lok Sabha sitting days per year, from the first session in 1952 to today. Every bar extracted directly from the Ministry of Parliamentary Affairs Statistical Handbook.

{% include ip-historical.html %}

*Peak: 168 days in 1954. The low points are not random — 1975–77 is the Emergency, 1977 is the election year, 2008 was 30 days (the lowest post-Emergency year), 2020 was COVID. The 10-year rolling average has not been above 100 since 1974.*

---

## Session by Session: The Full Picture

Every session of the 17th and 18th Lok Sabha — productivity, collapse points, and bills passed per sitting.

{% include ip-session-trend.html %}

*Budget 2023 is the worst single session: 33% overall, with Part II running at just 5%. Monsoon 2021 at 21% — Parliament was in session but almost entirely disrupted by the Pegasus and farm laws controversies. Budget 2022 at 123% shows the same parliament can work when it chooses to.*

---

## Five Years of the 17th Lok Sabha

These are aggregate statistics from the PRS summary of the complete 17th Lok Sabha (June 2019 – February 2024). They are not per-session estimates — they are the published totals.

{% include ip-17ls-stats.html %}

*The 729 Private Member Bills introduced → 2 discussed is not a rounding error. 729 bills were formally introduced by MPs; 2 received floor time. The last Private Member Bill to actually pass Parliament was in 1970.*

## MPs Who Raised Matters Parliament Never Addressed

Every year, MPs formally file matters they want to raise. When Parliament is disrupted, these go unaddressed. The chart counts exactly how many were filed but never taken up — under Rule 377 in Lok Sabha, and Rule 180 in Rajya Sabha.

{% include ip-unaddressed.html %}

*2022: 306 LS matters and 412 RS matters unaddressed. Winter 2022 alone accounts for 260 of the LS total — the session had 8 of 10 business days disrupted and 23 RS MPs suspended. The 2025 Budget figure (473 LS) is high because Rule 377 is deprioritised in busy sessions even when Parliament is functional — not a pure disruption signal. 2023 LS data absent from the Ministry's published dataset.*

---

## When Disruptions Hit the Two Houses Differently

The government tracks exact hours lost to forced adjournments per session, per house. Winter 2024 shows the sharpest contrast: Rajya Sabha lost 65.7% of its time to disruptions while Lok Sabha, in the same session, lost only 4.6%.

{% include ip-disruptions.html %}

*Winter 2024 RS was disrupted by a notice to remove the Vice-President and a judge impeachment notice — neither passed, but both consumed the house. LS continued functioning. Same Parliament, same weeks, opposite outcomes.*

---

## How India Compares

Counting parliamentary sitting days across countries is harder than it looks — different countries measure different things. The chart below uses only verified primary-source figures, with methodology noted per row.

{% include ip-global.html %}

*India's 55 days is an average across a full 5-year term. Japan's 150 days is calendar days the session is open, not necessarily plenary sittings. Germany and Canada figures are converted from sitting weeks (IPU data) — treating 1 week as 5 days. The comparison is directionally valid, not exact.*

---

## The Data

Every fact in this post is traceable to a primary source. Nothing is approximate.

<div style="display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0 1.5rem;">
  <a href="/assets/data/parliament_verified_facts.json" download style="display:inline-flex;align-items:center;gap:8px;padding:9px 16px;background:#111827;border:1px solid #1f2937;border-radius:7px;color:#e5e7eb;text-decoration:none;font-size:0.88em;font-family:system-ui,sans-serif;">
    <span style="font-size:1.1em;">⬇</span> JSON (62 verified facts with source URLs)
  </a>
  <a href="/assets/data/parliament_verified_facts.csv" download style="display:inline-flex;align-items:center;gap:8px;padding:9px 16px;background:#111827;border:1px solid #1f2937;border-radius:7px;color:#e5e7eb;text-decoration:none;font-size:0.88em;font-family:system-ui,sans-serif;">
    <span style="font-size:1.1em;">⬇</span> CSV (same 62 facts, flat format)
  </a>
</div>

**Primary sources used:**

| Claim | Source | URL |
|---|---|---|
| Both 2025 sessions — all figures | PRS Legislative Research | prsindia.org/parliamenttrack/vital-stats |
| 2025 session productivity (govt figures) | Press Information Bureau | pib.gov.in |
| 17th LS aggregate statistics | PRS Legislative Research | prsindia.org/parliamenttrack/vital-stats/functioning-of-the-17th-lok-sabha |
| Year-wise sitting days 1952–2022 | Ministry of Parliamentary Affairs Statistical Handbook 2023 | mpa.gov.in (PDF, Table 3) |
| Sitting days 2023–24 | Ministry of Parliamentary Affairs dataset #45 | data.gov.in (via dataful.in/datasets/45) |
| Australia 67 days/year since 1901 | Parliament of Australia, Practice7 | aph.gov.au |
| Japan 150-day ordinary session | IPU / Wikipedia citing official | data.ipu.org, en.wikipedia.org/wiki/National_Diet |
| Canada ~27 sitting weeks/year | IPU Parline | data.ipu.org/parliament/CA |
| Germany ≥20 sitting weeks/year | IPU Parline | data.ipu.org/parliament/DE |
| UK 2022–24 session sitting days | UK Parliament recess dates page | parliament.uk |

**What this post does not claim:**
All figures cited from secondary sources that cite PRS (e.g., The Week, Drishti IAS) are excluded from visualizations — only direct primary source fetches are used. Session productivity % for Monsoon 2020 and Special 2023 are unavailable (PRS pages returned 404).
