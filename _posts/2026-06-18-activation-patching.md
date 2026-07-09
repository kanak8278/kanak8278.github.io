---
title: "Activation Patching"
date: 2026-06-18
categories:
  - mechinterp-log
tags:
  - activation-patching
  - gemma
  - ioi
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "First activation patching run on Gemma produced a suspiciously clean result on the first try — which is exactly when to double check, not relax."
---

Built an [IOI](https://learnmechinterp.com/topics/ioi-circuit/)-style clean/corrupted pair on Gemma (name swap, same template) and ran
[activation patching](https://learnmechinterp.com/topics/activation-patching/) per [Wang et al., "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) — run corrupted, patch in a single clean activation,
measure logit-diff recovery. Took a while to get the direction straight in my head:
patching *into* the corrupted run *from* the clean run, asking "if this one piece were
clean, would the answer come back."

First heatmap across layers × positions produced a legible story — a small set of heads
at one layer recover most of the logit diff. Didn't expect it to be that clean on a first
try, which honestly made me a little suspicious of it rather than just pleased. The
method assumes the model's computation decomposes into independent swappable nodes, but
components share subspaces and are correlated with each other — patching one node can
silently patch in correlated information elsewhere too. A clean-looking result on the
first real attempt is exactly the situation where I should go check the corrupted prompt
wasn't accidentally doing something too easy, not the situation to stop looking.
