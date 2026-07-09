---
title: "Jacobian Lens"
date: 2026-06-16
categories:
  - mechinterp-log
tags:
  - jacobian-lens
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Jacobian lens generalizes logit lens via local linearization — more expensive than expected, and a nagging worry about picking whichever lens confirms the hypothesis."
---

[Jacobian lens](https://learnmechinterp.com/topics/jacobian-lens/) generalizes logit lens: instead of literally unembedding an intermediate
activation, take the local linearization of the actual downstream computation w.r.t. that
activation, $J = \partial \text{logits} / \partial h_l$. Logit lens implicitly assumes
everything after layer $l$ is just the unembed, which throws away real interaction
effects; Jacobian lens captures what the remaining layers actually do, at least to first
order.

More expensive than expected — full Jacobian per position isn't free, ended up
subsampling on Gemma. A few places where logit lens and Jacobian lens disagreed, which is
presumably the signal that later layers do non-trivial work on that direction rather than
just amplifying it.

Small nagging thought I keep having about this whole family of "lens" tools: there are now
several of them (logit lens, tuned lens, Jacobian lens) and no strong story for which one
is *right* for a given question, just tradeoffs. Easy to end up picking whichever lens
happens to produce the cleanest-looking story for the hypothesis I already had going in.
Not sure I have a fix for that other than being honest about it when it happens.
