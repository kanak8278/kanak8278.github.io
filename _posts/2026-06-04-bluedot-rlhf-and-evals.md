---
title: "BlueDot: RLHF, Constitutional AI, and dangerous capability evals"
date: 2026-06-04
categories:
  - mechinterp-log
tags:
  - ai-safety
  - bluedot
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Two days into BlueDot's Technical AI Safety intensive — daily 2hr calls plus readings. Training safer models and detecting danger sections: RLHF's actual objective, Constitutional AI, and what dangerous capability evals concretely test for."
---

Two days into [BlueDot's](https://bluedot.org/) Technical AI Safety intensive — daily 2hr calls plus a stack of shared readings in between. Deliberately structured differently from the MI curriculum, more breadth-first across the whole safety landscape rather than one deep technical thread.

"Training safer models" made the RLHF objective concrete in a way I'd only had loosely in my head before — you're optimizing

$$\mathbb{E}_{y \sim \pi_\theta}\left[ r_\phi(x, y) \right] - \beta \, D_{KL}\!\left(\pi_\theta(\cdot|x) \,\|\, \pi_{ref}(\cdot|x)\right)$$

reward from a learned preference model, penalized by KL divergence from the reference (pre-RLHF) policy so the model doesn't drift into reward-hacking degenerate text. Constitutional AI's variant — using AI feedback against a written constitution instead of a human preference model at the critique/revision stage — was new to me as a concrete mechanism rather than a buzzword.

"Detecting danger" walked through how the frontier labs actually structure evals — the concrete example that stuck was autonomous-replication-style evals (can a model, unassisted, get a copy of itself running on a new server, acquire compute, etc.) as a proxy for dangerous capability rather than trying to eval "danger" directly. Comparing Anthropic's, OpenAI's, DeepMind's, and Meta's public approaches side by side was more useful than I expected — the framing differs more than the actual techniques do.
