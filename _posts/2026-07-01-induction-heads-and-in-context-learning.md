---
title: "Induction heads and in-context learning"
date: 2026-07-01
categories:
  - mechinterp-log
tags:
  - induction-heads
  - in-context-learning
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Induction heads and the training-loss 'bump' — a mechanistic explanation for a loss curve shape I'd only ever explained behaviorally."
---

The part of Olsson et al. that actually changed how I think about something from my day
job: [induction heads](https://learnmechinterp.com/topics/induction-heads/) emerging is tied to a specific, sudden phase change in training loss
curves — the "bump," a brief plateau then a sharp drop, coincides with induction heads
forming. I've stared at loss curves with exactly that shape before without a mechanistic
story for why. Having a circuit-level explanation for a training-dynamics phenomenon I'd
only ever explained behaviorally is the first place this curriculum has paid off directly
against real work rather than feeling like a separate track.

Should say though — induction heads correlating with the bump isn't the same as induction
heads being the *whole* story for in-context learning generally. ICL shows up in settings
and scales where the clean induction-head story gets murkier, and I don't think that's
fully resolved in what I've read so far. Filing it as "a real mechanism for a real
phenomenon," not "the explanation for ICL."
