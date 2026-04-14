---
title: "The Fragmented Calendar: A Data Atlas of Global Stock Exchange Holidays 2025"
date: 2026-04-15
categories:
  - blog
tags:
  - finance
  - markets
  - data
  - visualization
  - india
  - global
toc: false
read_time: true
excerpt: "Japan loses more trading hours to holidays than any other major exchange. India doesn't have the most holidays — but it has the most fragmented ones. The pattern matters as much as the count."
---

<div style="border-left:3px solid #f59e0b; padding:12px 18px; background:rgba(245,158,11,0.06); border-radius:0 6px 6px 0; margin-bottom:2rem;">
<p style="margin:0; color:#d1b07a; font-size:0.96em; line-height:1.75;">
Japan loses <strong style="color:#f472b6">99 hours</strong> of trading time to holidays each year — more than any other major exchange. India loses <strong style="color:#f59e0b">87.5 hours</strong> across 14 closures. China loses only <strong style="color:#ef4444">72 hours</strong> despite having 18 holidays. Same counts, wildly different costs — because the pattern matters as much as the total. Every number below is verified from official exchange sources.
</p>
</div>

## Where Every Holiday Falls

Each dot is one weekday the market is closed. Position is exact — placed by week within the year. Hover to see the holiday name.

{% include smh-calendar.html %}

*India's row is a dashed line — one dot almost every month, never clustered. China's row is the opposite: six solid blocks you can plan around. Germany and LSE clear the entire mid-year with almost nothing.*

## What It Actually Costs

The count of holidays is misleading without session length. Germany and LSE each close 8 times but run 8.5-hour days. China closes 18 times but trades only 4 hours a day.

{% include smh-hours.html %}

*Japan's 2024 extension to 5.5-hour sessions (removing the lunch break, effective Nov 5 2024) pushed its annual hours-lost above everyone else — a fact absent from most market commentary.*

## What Each Country Decides Is Worth Stopping For

Japan closes for 13 civic holidays and only 2 cultural ones. India is the opposite — 10 religious festivals and 3 civic days. The breakdown reveals institutional character more than any count does.

{% include smh-types.html %}

*JPX has a category none of the others have: 3 exchange-declared market holidays (Jan 2, Jan 3, Dec 31) with no civic or religious basis — the exchange simply decided those days don't trade.*

## How Many Weeks Are Actually Interrupted

Same holiday count can mean very different disruption rhythms. Each square below is one trading week — colored if it contains at least one closure, dark if completely clean.

{% include smh-weeks.html %}

*China has 18 holidays but only 9 disrupted weeks — clusters compress the damage. India has 14 holidays but 12 disrupted weeks because every closure is isolated. Xetra's 33-week uninterrupted streak (late April through November) is the cleanest continuous trading window of any major exchange.*

---

## The Data

Every date, name, type, and block grouping — verified from official exchange circulars and the open-source [`exchange_calendars`](https://github.com/gerrymanoim/exchange_calendars) package. No approximations.

<div style="display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0 1.5rem;">
  <a href="/assets/data/exchange_holidays_2025.json" download style="display:inline-flex;align-items:center;gap:8px;padding:9px 16px;background:#111827;border:1px solid #1f2937;border-radius:7px;color:#e5e7eb;text-decoration:none;font-size:0.88em;font-family:system-ui,sans-serif;">
    <span style="font-size:1.1em;">⬇</span> JSON (full dataset + sources)
  </a>
  <a href="/assets/data/exchange_holidays_2025.csv" download style="display:inline-flex;align-items:center;gap:8px;padding:9px 16px;background:#111827;border:1px solid #1f2937;border-radius:7px;color:#e5e7eb;text-decoration:none;font-size:0.88em;font-family:system-ui,sans-serif;">
    <span style="font-size:1.1em;">⬇</span> CSV (92 rows, one per exchange-holiday)
  </a>
</div>

**Sources per exchange:**

| Exchange | Holiday source | Hours source |
|---|---|---|
| NSE / BSE | NSE official circular (nseindia.com/resources/exchange-communication-holidays) | NSE official market timings |
| NYSE / NASDAQ | NYSE/ICE press release Dec 2022 + SEC filing Jan 2025 (Carter mourning day) | NYSE.com/trade/hours-calendars |
| LSE | londonstockexchange.com/equities-trading/business-days | Confirmed: 08:00–16:30 daily |
| JPX | jpx.co.jp/english/corporate/about-jpx/calendar | JPX press release Nov 3 2024 — extension to 12:30–15:30 |
| HKEX | hkex.com.hk/services/trading/derivatives/overview/trading-calendar-and-holiday-schedule | 9:30–12:00 + 13:00–16:00 |
| SSE | english.sse.com.cn/start/trading/schedule | 9:30–11:30 + 13:00–15:00 |
| Xetra | Deutsche Börse 2025 calendar PDF (cashmarket.deutsche-boerse.com) | "9:00–17:30 CET" — official |
