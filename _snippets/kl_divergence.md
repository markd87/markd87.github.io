---
layout: post
title: "KL divergence"
date: 2020-11-09
---

Kullback–Leibler (KL) divergence is a "distance" measure between two probability distributions $P(x)$ and $Q(x)$ defined by:

$$
D_{KL}(P\parallel Q) = \int_{-\infty}^{\infty} P(x)\log \left( \frac{P(x)}{Q(x)} \right)dx.
$$

The reason it's not a formal distance is that it's not symmetric 
$D_{KL}(P\parallel Q) \ne D_{KL}(Q\parallel L)$.

$D_{KL}(P\parallel Q) \ge 0$, with equality if and only if $P(x)=Q(x)$.

From the definition, the distance is the expectation value of the difference of logs of the two distributions.

KL is commonly used in bayesian inference where a distribution $Q(x)$ is trying to approximate the true distribution $P(x)$ using $D_{KL}(Q||P)$ as the distance measure (note the order).
When $Q(x)>0$, the difference between the two distances should be minimal, in particular $P(x)>0$, and when $Q(x)=0$ the KL distance is zero anyway because the weight is zero. 
This is known as zero-forcing, as it results in places with $Q(x)=0$ even if $P(x)>0$.