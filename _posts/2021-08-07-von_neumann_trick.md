---
layout: post
title: "Von Neumann Trick"
date: 2021-08-07
---

How to turn a biased coin into an unbiased coin?

A biased coin means the probability of either heads (H) or tails (T) is $p>0.5$.
The trick is to flip the coin twice, setting HT to "H" and TH to "T", and if the coin comes HH or TT ignore and try again. This way the probability of "H" and "T" are now equal to $p(1-p)$ making it an unbiased coin.