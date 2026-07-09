---
title: "BlueDot: interpretability in practice, and wrapping up"
date: 2026-06-08
categories:
  - mechinterp-log
tags:
  - ai-safety
  - bluedot
  - mechinterp
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Finished the BlueDot intensive. The interpretability-in-practice section was the one place the two tracks actually touched — SAE features as a monitoring layer, not just an explanation tool."
---

Finished the BlueDot Technical AI Safety intensive today. [Certificate here](https://bluedot.org/certification?id=recI4k6UjU2FbeSKO).

The "Understanding AI" section — specifically interpretability in practice — was the one place this course actually touched the MI curriculum directly. The framing that stuck: interpretability isn't just post-hoc explanation, it's pitched as an additional monitoring layer that doesn't rely on the model's own outputs being honest — e.g. using SAE features to flag when a model's internals look like they're representing something ("this looks like deception-related activity") independent of what it says. Behavioral evals and interpretability as two mostly-independent lines of defense rather than one subsuming the other. Good to have that framing before getting into SAEs properly later in the MI curriculum.

Broader course was useful context but a real gear-change from the technical depth of the last two weeks — good to have both the "why does any of this matter, what's the actual threat model" grounding and the narrow technical skill separately, rather than only ever doing one. Back to the MI curriculum tomorrow — Interpretability Fundamentals next.
