---
title: "Logit Lens, Direct Logit Attribution, Gemma setup"
date: 2026-06-13
categories:
  - mechinterp-log
tags:
  - logit-lens
  - dla
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Setting up TransformerLens on gemma-2-2b (RMSNorm complications), then logit lens and Direct Logit Attribution — both fast since it's just residual stream linearity."
---

Set up [TransformerLens](https://learnmechinterp.com/topics/transformerlens/) on gemma-2-2b today, mostly fighting the harness rather than
learning anything conceptually new. Gemma uses RMSNorm, not LayerNorm — the standard logit
lens writeup assumes LayerNorm's mean-centering, had to double check the RMSNorm version
doesn't silently break the interpretation.

Read [nostalgebraist's original Logit Lens post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) first — [logit lens](https://learnmechinterp.com/topics/logit-lens-and-tuned-lens/): $\text{logits}_l = W_U \cdot \text{LN}(h_l)$, unembed an intermediate residual
stream as if the model stopped there. [DLA](https://learnmechinterp.com/topics/direct-logit-attribution/) is the natural extension, and it's really just
Anthropic's ["Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) applied — residual stream is a
sum of per-component writes, freeze the final norm's scale, and each head/MLP's
contribution to the final logit becomes a dot product with $W_U$, additive across
components. Both landed fast, it's just linearity of the residual stream.

Got basic DLA running on an IOI-style prompt on Gemma tonight. Numbers look sane.

Todo:
- sanity check DLA against a real ablation baseline, not just "looks reasonable"
- the "freeze the norm scale" step in DLA is itself an approximation — check how much it
  actually varies between prompts before trusting the additive story too much
