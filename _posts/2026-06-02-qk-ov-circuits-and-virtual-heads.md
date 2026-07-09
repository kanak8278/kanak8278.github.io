---
title: "QK/OV circuits and virtual heads"
date: 2026-06-02
categories:
  - mechinterp-log
tags:
  - transformers
  - fundamentals
layout: blog-post
toc: false
read_time: false
related: false
excerpt: "Came back after a couple days off. LayerNorm was fine, but QK/OV circuits and composition/virtual attention heads took a lot more time than expected to actually get intuition for."
---

Didn't get back to this over the weekend, picking it up again today. [LayerNorm](https://learnmechinterp.com/topics/layer-normalization/) was fine, mostly folding into intuitions I already had.

[QK/OV circuits](https://learnmechinterp.com/topics/qk-ov-circuits/) and [composition/virtual attention heads](https://learnmechinterp.com/topics/composition-and-virtual-heads/) were a different story — took a lot more time than I expected given everything else in this section was mostly review. The framing of attention as a QK circuit (where to look) and an OV circuit (what to write) decoupled from each other, and then heads *composing* across layers to form effective "virtual" heads that don't correspond to any single attention head — none of that was intuitive on first pass. A lot of the standard explanations lean on simplifications (ignoring LayerNorm's effect on the circuits, treating the residual stream as a clean linear channel) that make the math tractable but also make it feel like the picture is glossing over something. Had to work through it a few times before the composition intuition actually stuck, and I'm still not 100% sure I'd explain virtual heads well to someone else.

Finished the from-scratch PyTorch transformer implementation today too. Feels like a reasonable place to pause before the BlueDot AI Safety intensive starts tomorrow — going to be light on mech interp for the next several days.
