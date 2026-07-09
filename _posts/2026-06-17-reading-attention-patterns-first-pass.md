---
title: "Reading Attention Patterns (first pass)"
date: 2026-06-17
categories:
  - mechinterp-log
tags:
  - attention-patterns
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "First cold read of Gemma's attention patterns, mostly noise without a hypothesis to guide it."
---

Pulled up [attention patterns](https://learnmechinterp.com/topics/reading-attention-patterns/) on Gemma with circuitsvis for a handful of prompts, no
hypothesis going in. Mostly noise. What I could actually tell apart:

- a few heads: strict previous-token attention
- a few heads: strict first-token / attention-sink behavior
- most heads: nothing legible from eyeballing alone

Suspect this'll make a lot more sense once patching tells me which heads matter for
something specific, instead of going in blind.
