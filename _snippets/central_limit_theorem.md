---
layout: post
title: "Central Limit Theorem intuition"
date: 2021-02-14
---

A nice intuition for the central limit theorem I've recently read about has to do with maximum entropy.

The central limit theorem says that the sum of independent random variables tends to a Gaussian distribution with given means and standard deviation.

For a given class of distributions (e.g. finite known n-moments), a [maximum entropy distribution](https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution) is the distribution with least prior-information, where the entropy is defined as, 
$$
H(X) = -\int_{-\inf}^{\inf} p(x)\log p(x) dx.
$$

The maximum entropy distribution with finite mean and variance is the **Gaussian distribution**.

Therefore if a random variable is the average of multiple independent random variables, then we know the mean and variance but not any other moments, therefore within this class, the distribution with highest entropy or most uncertainty or least prior-information is the Gaussian distribution.
