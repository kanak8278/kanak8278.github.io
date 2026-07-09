---
title: "Induction heads, K-composition"
date: 2026-06-29
categories:
  - mechinterp-log
tags:
  - induction-heads
  - path-patching
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Confirmed the K-composition claim with path patching rather than just eyeballing attention patterns — the first real circuit-level causal claim in this whole curriculum."
---

Spent today confirming the *composition* claim for [induction heads](https://learnmechinterp.com/topics/induction-heads/) rather than just the attention pattern —
checking the induction head's key genuinely reads the previous-token head's output, not
just correlates with it. Path patched from the previous-token head's output into the
induction head's K input specifically. Result held: patching that one path kills the
induction behavior almost entirely, other paths into the same head don't. First time in
this whole curriculum I've produced a real circuit-level causal claim instead of an
observation.

One caveat I don't want to lose: this is one head, on one synthetic repeated-token
distribution. It doesn't establish that this circuit is doing the same thing on messier
natural language, or that "induction head" is a clean natural kind rather than a label I'm
applying to whatever pattern-matches the definition closely enough. The field talks about
circuits generalizing across prompts and models more confidently than any single result
like this actually earns.
