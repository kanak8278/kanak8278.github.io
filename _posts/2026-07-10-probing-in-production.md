---
title: "Probing in production, and where this leaves things"
date: 2026-07-10
categories:
  - mechinterp-log
tags:
  - probing
  - reflection
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Closing out everything before steering: probing in production, and an honest note about selection effects in what counts as a clean circuit-finding story."
---

Last piece of the probing block: [how any of this gets used in production](https://learnmechinterp.com/topics/probes-in-production/) as a lightweight real-time
monitor rather than a one-off research artifact — cheap linear probes running alongside
inference, flagging when an internal direction lights up, versus expensive full evals.
Trained a probe on Gemma and clocked it against a full forward pass; overhead is close to
nothing, which is the whole point.

That closes out everything before steering. Looking back, the throughline for this
stretch has been the shift from behavioral claims to actual causal ones — patching, path
patching, causal abstraction, self-repair — that's the part that didn't exist anywhere in
my prior ML background and is clearly the actual core skill here.

Honest overall note before moving on: a lot of what "worked" in this stretch worked on
IOI-style toy tasks and repeated-token diagnostics specifically because they're clean and
legible enough to *make* a circuit-finding story work. That's probably the right way to
learn the tools, but I'd be surprised if the same techniques come apart less cleanly on
messier, more realistic capabilities — and most of what I've read so far doesn't dwell on
the cases where the method was tried and didn't produce a clean story. Selection effect
worth remembering going into steering, not a reason to distrust the tools themselves.
