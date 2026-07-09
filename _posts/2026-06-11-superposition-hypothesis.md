---
title: "Superposition Hypothesis"
date: 2026-06-11
categories:
  - mechinterp-log
tags:
  - interpretability-fundamentals
  - superposition
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Anthropic's Toy Models of Superposition reframes capacity as combinatorial, not additive — though the jump from toy model to real LLM is an extrapolation nobody's directly shown."
---

First real wall — [Superposition Hypothesis](https://learnmechinterp.com/topics/superposition/). Spent most of the day on Anthropic's ["Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html).

- $d$-dim layer represents more than $d$ features as near-orthogonal directions
- tolerable when features are sparse, interference cost $\sum_{i \neq j}(W_i \cdot W_j)^2$
  paid rarely enough that ReLU cleans up the noise on the read side
- reconstruction loss vs sparsity curves in the toy model make the tradeoff concrete

Reframes capacity as combinatorial instead of additive — most of my intuitions about what
a given hidden size "can hold" were quietly assuming no superposition at all.

That said, the toy model is a tiny synthetic ReLU-output autoencoder with a hand-picked
feature distribution — the jump from "this happens in a toy setup we built specifically to
show it" to "this is what's actually happening inside a 2B-parameter language model" is an
extrapolation, not something anyone's shown directly. And there's a deeper issue underneath
all of this that nobody seems to fully resolve: "feature" doesn't have a definition that
exists independent of the tools used to find one. There's no ground-truth feature list to
check the toy model or an SAE against. Might just be the nature of a young field, but it
means a lot of claims here are more "self-consistent" than "verified."
