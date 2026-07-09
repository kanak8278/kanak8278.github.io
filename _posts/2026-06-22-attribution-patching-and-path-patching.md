---
title: "Attribution Patching and Path Patching"
date: 2026-06-22
categories:
  - mechinterp-log
tags:
  - attribution-patching
  - path-patching
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Attribution patching as a cheap linear approximation of real patching, and path patching to isolate causal paths through specific components."
---

[Attribution patching](https://learnmechinterp.com/topics/attribution-patching/) ([Neel Nanda's writeup](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)): linear (gradient) approximation of the patching effect,
$\text{effect} \approx (a_\text{clean} - a_\text{corr}) \cdot \nabla_a \mathcal{L}$, one
backward pass covers every node instead of one forward pass per node. Ran it on Gemma,
cross-checked against Thursday's real activation-patching heatmap:

- mostly agreed
- a few nodes where the linear approximation clearly broke down — large nonlinear effect,
  gradient at the corrupted point doesn't represent it well

Good to see it fail somewhere. Would've trusted it less if it had matched everywhere.

Path patching: patch only the effect flowing through one specific downstream path,
holding everything else fixed. This is what actually lets you claim "head A affects the
output *via* head B" instead of just "head A matters somehow" — a meaningfully stronger
claim than plain activation patching gives you.
