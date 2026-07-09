---
title: "Probing: classifiers and truthfulness"
date: 2026-07-06
categories:
  - mechinterp-log
tags:
  - probing
  - truthfulness
  - gemma
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Linear probes on Gemma for sentiment and truthfulness, and the non-identifiability problem underneath 'the truth direction.'"
---

Trained [linear probes](https://learnmechinterp.com/topics/probing-classifiers/) on Gemma's residual stream to test what's linearly decodable at
each layer, starting with sentiment before touching [truthfulness probing](https://learnmechinterp.com/topics/truthfulness-probing/). The
methodological catch that's easy to miss coming from standard ML eval habits: high probe
accuracy only shows the information is linearly present, not that the model *uses* it for
anything downstream.

[Burns et al.'s "Discovering Latent Knowledge"](https://arxiv.org/abs/2212.03827) motivation stuck with me — they specifically
avoid labeled truthfulness data because supervised probes can key onto spurious correlates
of the label instead of the concept itself. There's a sharper version of this problem too:
multiple different linear directions can often achieve near-identical probe accuracy on
the same data. "The truth direction" as singular and well-defined is doing more work in a
lot of writeups than the non-identifiability of the probe direction really supports.

Getting close to the end of this stretch of the curriculum now, which got me thinking about actually applying to [ARENA](https://learnmechinterp.com/topics/arena/) this year rather than just working through material on my own indefinitely — structured feedback on this would help a lot more than another few weeks of solo reading.
