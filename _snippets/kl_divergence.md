---
layout: post
title: "KL divergence"
date: 2020-11-09 23:28 +0000
---

Kullback–Leibler (KL) divergence is a "distance" measure between two probability distributions $P(x)$ and $Q(x)$ defined by:

$$
D_{KL}(P\parallel Q) = \int_{-\infty}^{\infty} P(X)\log \left( \frac{P(x)}{Q(x)} \right).
$$

The reason it's not a formal distance is because it is not symmetric 
$D_{KL}(P\parallel Q) \ne D_{KL}(P\parallel Q)$.

From the definition, the distance is the expectation value of the difference of logs of the two distributions.
