---
layout: post
title: "Friendship Paradox"
date: 2020-10-03 20:58:00 +0100
tags: thoughts
---

The friendship paradox is a seemingly paradoxical observation about social-like networks, which says that on average a person in the network has fewer friends than his friends have. First derived by Scott L. Feld [<a href="http://www.macroeconomics.tu-berlin.de/fileadmin/fg124/networks/Lectures/Summer2012/Material/American_Journal_of_Sociology_1991_Feld.pdf" target="_blank">paper</a>].

It seems counter-intuitive because it's not clear how the symmetry is broken between a person and their friends, friendship being a symmetric relationship. The intuition for this is the fact that by looking at ones friends we are biasing the sample towards people that tend to have friends, and therefore likely to have more friends.
Many applicable examples exist, one such is citations. In academia people cite each other, however for a random researcher it is more likely that the people they cite have more citations than the random researcher.

Interestingly, this paradox has important consequences for disease spreading, in particular the kind that spreads in clusters, where one or few people infect a large number of other people. In such an infection network any one person who's infected is likely to have been infected by someone else as opposed to being the original infector or the super-spreader. Therefore, to obtain control over such a spreading mechanism, it is more beneficial to identify not who's infected (you) but who infected you (your friends), i.e. trace back your infection eventually to the cluster root.

For my own reference, I wanted to go over the proof available on [Wikipedia](https://en.wikipedia.org/wiki/Friendship_paradox).

We start with a graph $G(E,V)$ with $E$ edges representing friendships, and $V$ vertices representing people, two people are friends if there's an edge connecting them.

The first quantity is the average number friends of a random person, denoted $\mu$. This is the average degree of a vertex, denoted $d(v)$. Since there are V vertices, the average is,

$$
\mu = \frac{\sum_{v\in V} d(v)}{V} = \frac{2E}{V}
$$

the second equality comes from summing all the edges from each vertex, but since each edge connects two vertices, we count the edges twice.

The next quantity is the average number of friends a friend has.
Randomly selecting a friend of a random person means randomly selecting an edge which has two friends and then selecting one of the vertices as the friend. For a chosen vertex $v$, there are $d(v)$ edges out of $E$ edges in the graph, therefore the probability of choosing that $v$ is $d(v)/E$. Then, selecting the particular edge, is a random selection from 2 vertices, given a factor of half. In total the probability is $\frac{d(v)}{2E}$
The average number of friends of a friends of a random person is then the expectation value of the degree $d(v)$,

$$
\nu = \sum_v \frac{d(v)}{2E}d(v) = \sum_v \frac{d(v)^2}{2E}.
$$

Now we need to massage this last expression by noting the definition of the variance of the vertex degree $d(v)$ in the graph. This is given by:

$$
\sigma^2 = <d(v)^2> - \mu^2 =\frac{\sum_v d(v)^2}{V} - \mu^2.
$$

Rewriting this as,

$$
\frac{\sum_v d(v)^2}{V} = \sigma^2 + \mu^2,
$$

We can write for the average number of friends of a friend as (multiplying and dividing by $V$ the expression for $\nu$):

$$
\nu = \frac{V}{2E} (\mu^2 + \sigma^2) = \mu + \frac{\sigma^2}{\mu}.
$$

All quantities being positive, we get that $\nu > \mu$, i.e. for a random person, the average number of friends a typical friend has is greater than the random person's number of friends.
Put simply, our friends have more friends than we do, on average.
