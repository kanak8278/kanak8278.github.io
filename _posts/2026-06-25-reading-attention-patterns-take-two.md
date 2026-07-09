---
title: "Reading Attention Patterns, take two"
date: 2026-06-25
categories:
  - mechinterp-log
tags:
  - attention-patterns
  - path-patching
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Re-reading the same attention patterns from over a week ago, this time with a hypothesis from patching in hand — completely different experience."
---

Went back to the same [attention pattern](https://learnmechinterp.com/topics/reading-attention-patterns/) visualizations on Gemma from over a week ago, this time only looking
at heads path patching had flagged. Completely different experience — patterns that
looked like noise before are obviously doing something specific once you know what to
look for. One flagged head cleanly attends name-token → previous occurrence of that name,
exactly the "copying" behavior the patching result implied. Reading attention patterns
cold isn't a great use of time; reading them with a hypothesis in hand is a different tool
entirely.
