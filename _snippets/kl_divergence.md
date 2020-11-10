---
layout: post
title: "KL divergence"
date: 2020-11-09 23:28 +0000
---

Kullback–Leibler (KL) divergence is a "distance" measure between two probability distributions $P(x)$ and $Q(x)$ defined by:

$$
D_{KL}(P\parallel Q) = \int_{-\infty}^{\infty} P(x)\log \left( \frac{P(x)}{Q(x)} \right)dx.
$$

The reason it's not a formal distance is that it's not symmetric 
$D_{KL}(P\parallel Q) \ne D_{KL}(Q\parallel L)$.

$D_{KL}(P\parallel Q) \ge 0$, with equality if and only if $P(x)=Q(x)$.

From the definition, the distance is the expectation value of the difference of logs of the two distributions.

Another preferable property is the zero-avoiding, when comparing a distribution $Q(x)$ to $P(x)$, if $Q(x)$ has zero value for $x$ where $P(x)$ is non-zero, the KL divergence becomes infinite.
