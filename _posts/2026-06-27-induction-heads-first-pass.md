---
title: "Induction heads, first pass"
date: 2026-06-27
categories:
  - mechinterp-log
tags:
  - induction-heads
  - circuit-finding
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "First pass at induction heads following Olsson et al. — found textbook induction patterns on Gemma via a repeated-token diagnostic."
---

Started [induction heads](https://learnmechinterp.com/topics/induction-heads/) / circuit finding, following [Olsson et al., "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). Mechanism is a clean
two-head composition:

- previous-token head writes "the token before me was X" into the residual stream
- induction head K-composes with that write, searches for prior occurrences of the
  *current* token, copies whatever followed it last time
- `[A][B]...[A] → [B]`

Used a repeated-random-token sequence (the standard diagnostic) instead of IOI prompts —
much cleaner signal, a handful of heads have textbook induction attention patterns as soon
as I plotted per-head scores across layers.
