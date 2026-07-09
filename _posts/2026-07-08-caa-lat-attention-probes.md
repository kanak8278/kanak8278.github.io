---
title: "CAA, LAT, attention probes"
date: 2026-07-08
categories:
  - mechinterp-log
tags:
  - probing
  - caa
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "CAA, LAT, and attention probes — CAA previews the steering block ahead, the other two felt less mature."
---

- **[CAA](https://learnmechinterp.com/topics/caa-method/)** — mean difference between paired contrastive prompts (same prompt, truthful vs
  untruthful completion) gives a direction that's both a probe and, later, an intervention
  vector. Most intuitive of the three, and it previews the steering block instead of
  feeling isolated from it.
- **[LAT](https://learnmechinterp.com/topics/lat-probing/)** (linear artificial tomography) — fiddlier than the writeups suggested, sensitive
  to exactly which token positions / layers get aggregated. Spent more time on
  hyperparameter-adjacent choices than on the actual concept.
- **[Attention probes](https://learnmechinterp.com/topics/attention-probes/)** — probing attention patterns/outputs directly instead of the
  residual stream. Felt like the least mature of the three, thinner track record in what
  I read.
