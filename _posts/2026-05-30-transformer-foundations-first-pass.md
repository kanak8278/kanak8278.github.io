---
title: "Transformer foundations, first pass"
date: 2026-05-30
categories:
  - mechinterp-log
tags:
  - transformers
  - fundamentals
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Breadth-first pass through the transformer basics — mostly review given the DL background, so moved fast. Started implementing a transformer from scratch in PyTorch."
---

Starting properly today. Going breadth-first through the fundamentals section first: [prerequisites](https://learnmechinterp.com/topics/mi-prerequisites/), [architecture intro](https://learnmechinterp.com/topics/transformer-architecture/), [attention mechanism](https://learnmechinterp.com/topics/attention-mechanism/), [MLPs](https://learnmechinterp.com/topics/mlps-in-transformers/), [decoding strategies](https://learnmechinterp.com/topics/decoding-strategies/). Most of this is genuinely review — nothing here that a few years of DL work hasn't already covered, so moved through it quickly rather than reading every word carefully.

Started implementing a transformer from scratch in PyTorch alongside the reading, mostly to make sure nothing's rusty and to have something to poke at once the interpretability-specific stuff starts. Attention and MLP blocks straightforward. Left LayerNorm, QK/OV circuits, and composition/virtual heads for next time — skimmed them today and they're clearly denser than the rest, want to actually sit with them properly instead of rushing.
