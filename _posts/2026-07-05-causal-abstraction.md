---
title: "Causal Abstraction"
date: 2026-07-05
categories:
  - mechinterp-log
tags:
  - causal-abstraction
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Causal abstraction formalizes what patching has been informally assuming all along — and most practice, including my own, doesn't actually run the rigorous version."
---

[Causal abstraction](https://learnmechinterp.com/topics/causal-abstraction/) formalizes the question patching has been answering informally this
whole time: does a high-level causal variable I care about ("this head encodes the
sentence's subject") actually correspond to a specific low-level mechanism, not just
correlate with behavior? Tool is the interchange intervention — swap only the value of the
proposed high-level variable between two runs, check whether the low-level model behaves
exactly as the high-level causal model predicts, no more and no less.

Useful to have this named properly — it's the rigor check for everything the last two
weeks of patching work has been implicitly assuming. Redwood Research's ["Causal Scrubbing"](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) sequence is the practical version of this — actually implementable, not just a
framework paper. Also mildly deflating: most of the
patching-based circuit-finding work I've been reading (including my own from the last
couple weeks) doesn't actually run this full rigorous check, it eyeballs a heatmap and
calls it a circuit. The rigorous version exists, most practice doesn't reach for it.
