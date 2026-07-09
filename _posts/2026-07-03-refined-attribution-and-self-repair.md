---
title: "Refined attribution methods and self-repair"
date: 2026-07-03
categories:
  - mechinterp-log
tags:
  - attribution-patching
  - self-repair
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Refined attribution methods, and self-repair — a genuinely surprising result that maps onto failure-compensation patterns in multi-agent systems, with an open question about whether it's real or an ablation artifact."
---

[Refined attribution methods](https://learnmechinterp.com/topics/refined-attribution-methods/) patch attribution patching's linear-approximation errors from
Jun 22 — closer to integrated gradients (path-integrating rather than single-point
gradient) to account for the actual nonlinearity between clean and corrupted.

[Self-repair](https://learnmechinterp.com/topics/self-repair/) is the more interesting result — read [Rushing & Nanda, "Explorations of Self-Repair in Language Models"](https://arxiv.org/abs/2402.15390) for this one. Ablate an important head, and other
components — often specific "backup" heads doing near-nothing normally — partially
compensate, sometimes within the same forward pass. Genuinely surprising the first time I
saw it on Gemma (ablated the top-flagged head, expected the logit diff to collapse, it
dropped much less than predicted). Reminded me immediately of failure-compensation
patterns in multi-agent systems — redundant capacity that only activates once the primary
path is gone — except here it's inside one forward pass of one model.

Genuine open question I don't think has a clean answer yet: is this "repair" a real
functional property of the trained network, or partly an artifact of how the ablation
itself is done? Mean-ablating a head pushes the model into an out-of-distribution
activation state it never sees during training — some of what looks like "repair" could
be the model just behaving weirdly under an input it's never encountered, not a genuine
robustness mechanism. Not sure how you'd fully separate those two stories.
