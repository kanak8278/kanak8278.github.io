---
title: "What is Interpretability? / Linear Representation Hypothesis"
date: 2026-06-09
categories:
  - mechinterp-log
tags:
  - interpretability-fundamentals
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "The Linear Representation Hypothesis clicked fast via word2vec-style intuitions — though it's a hypothesis with known non-linear counterexamples, not a settled fact."
---

["What is Interpretability?"](https://learnmechinterp.com/topics/what-is-mech-interp/) was mostly framing — behavioral vs mechanistic, the bet that
network internals are structured enough to be worth reverse-engineering at all. Went fast,
this is methodology not technical content.

[Linear Representation Hypothesis](https://learnmechinterp.com/topics/linear-representation-hypothesis/) landed immediately, for a slightly annoying reason: it's
just the word2vec analogy intuition (`king - man + woman ≈ queen`) generalized and taken
seriously as a claim about *all* features, not a party trick. Concepts as directions
$v_c$ in activation space, present via $x \cdot v_c$.

Worth flagging though — this is a hypothesis, not an established fact, and I don't think
people always present it that way. There are already known counterexamples (circular /
periodic features like days-of-week or modular arithmetic representations that live on a
low-dim curve, not a line) and it's not obvious how much of the network's real computation
lives in that non-linear leftover. Convenient that linear directions are the easy thing to
probe for — not sure that's the same as them being the main thing that's there.
