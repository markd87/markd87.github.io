---
layout: post
title:  "Power of estimations"
date:   2019-07-01 17:03:47 +0100
categories: posts
tags: physics science
---

One of the fastest ways to solve any problem is to know the answer in advance. 
Especially in Physics, one of the most important tools to have is the ability to estimate the answers to problems before actually solving them. This involves invoking physical or general considerations which allow to get an order of magnitude estimate to a given problem be it physics related or completely general. Sometimes this practice is also
called back of the envelope calculation.
The greatest Physicists were famous in their ability to make such back of the envelope estimations of various problems,
most famous of which were Fermi, Feynman and Landau.

During my studies I took a course on methods of estimations and approximations taught by Prof. Nir Shaviv. In the first lecture
he said that Physicists are like Superman, they can solve any problem.
Here I wanted to present the famous Fermi problem which shows how one can tackle a seemingly impossible problem using the power of estimations,and then I give a short example of the methods of estimation used in the context of a Physics problem.

The question Fermi used to present as the first problem was: "How many piano tuners are there in Chicago?"
The approach to this and other seemingly impossible problems is through separating it to smaller problems which we
can estimate more easily the answer to and then construct the final answer, also known as 'divide and conquer'.
The sub-problems that need to be solved are: 
the number of people in Chicago: This is definitely more than 1 million and less than 100 million, we'll take 10 million.
Frequency of piano tunings: definitely more than once a month and more than once per decade, we'll take once a year. 
It takes about 2 hours to tune a piano (88 keys). Number of pianos: Here there are a number of ways to estimate, with various resolutions. We can assume 3 people per household on average with 5% of households having a piano giving: 160,000 pianos. 
An alternative way is to consider how many people hold a piano, there are about 10% (less than 100% more than 1%) who play the piano
but probably much less hold one, about 10% of them will also hold a piano, so we'll take $$ 1\times 10^{-2} $$. In addition there are pianos usually in churches and schools, these are
about 1 per 1000 people, so additional $$ 2\times10^{-3} $$, giving a total of 120,000 pianos. We can take the average of the two
as the final estimate: 140,000 pianos.
The final estimation is the amount of time a piano tuner works: we take a standard of 5 days a week, 8 hours a day, 50 weeks a year,
which is 2000 piano tuning hours a year. At 2 hours per piano this gives 1000 piano tunings per year.
Since there are 140,000 pianos to be tuned per year, there should be about 140 piano tuners.
With a tool such as <a href='http://www.wolframalpha.com/input/?i=how+many+piano+tuners+are+there+in+Chicago'>Wolfram Alpha</a>
This question can actually be answered exactly and quickly, giving ~290, which is of the same order of magnitude as our result. One
can play with the estimates to see where the estimations can be improved.

A more sophisticated example of Fermi's brilliant use of the power of estimation is the estimation of the energy released in the
Trinity atomic bomb test in 1945 as part of the Manhattan project leading to the bombing of Hiroshima and Nagasaki less than a month later.
The way he estimated it required a little more than just number guessing but some actual data which he obtained by witnessing the
explosion and the resulting shock wave itself. By throwing pieces of paper where he stood, he measured the displacement outwards and inwards
of these pieces papers due to the passing shock front. He found that they were displaced by 2.5m, and he stood 16km from the place of explosion.
The way to estimate the total energy is using conservation of energy and the work done by the shock wave on the air at the place where Fermi stood.
The volume displaced is the hemispherical shell of thickness 2.5m: $$ \Delta V=(2\pi r^2)d=4\times10^9 m^3 $$ (2 because only half a sphere).
The work done is $$ E=W=P\Delta V $$, the pressure here is the over pressure above atmospheric pressure. 1 atmosphere is
$$ 10^5N/m^2 $$, which is like 10 tons of weight on a human body, this is not reasonable, since in that case Fermi wouldn't be able to 
withstand it. It is probably more than $$ 10^2 N/m^2 $$ or 10kg, so we can take $$ P=10^3 N/m^2 $$, giving: $$ E=4\times10^{12} J= 1 KT $$.
This is not the total energy, since not all the energy goes into the kinetic energy of the shock, some goes to light, nuclear radiation, 
heating and more. So an additional factor of a few, say 4, gives an estimate for the total energy of $$ 4 KT $$.

Another physicist who used the power of estimation to estimate the energy released in the explosion, using a different 
method and different available data, was <a href='https://en.wikipedia.org/wiki/G._I._Taylor'>Taylor</a> who in 1950 using a set of photos released to the media showing the blast wave
expansion, conveniently giving the time elapsed and the distance scales on the photos. Taylor used the method of dimensional analysis to
estimate this energy. In any dimensional analysis problem, one looks for dimensionless parameters constructed from the relevant physical
parameters in the problem. The parameters he used were $$ E $$ - total energy, $$ \rho $$ - external density, $$ t $$ - time elapsed, $$ r $$ - radius of expansion.

$$\begin{eqnarray} 
 [E] &=& M L^2 S^{-2} \\
 [\rho] &=& M L^{-3} \\
 [t] &=& S \\
 [r] &=& L
\end{eqnarray}$$

We need to find the exponents which will give a dimensionless quantity:

$$\begin{eqnarray} 
	E^{a}\rho^{b}t^{c}r^{d} &=& const,  \\
	M^{a}L^{2a}S^{-2a}M^bL^{-3b}S^{c}L^d &=& M^{a+b}L^{2a-3b+d}S^{c-2a}
\end{eqnarray}$$

Therefore,

\$$\begin{eqnarray} 
	a+b=0, \\
	2a-3b+d=0, \\
	c-2a=0 \\
\end{eqnarray}$$

The answer doesn't have to be unique, since we can take the constant to any power.
We get

\$$\begin{eqnarray} 
	a=-b,
	d=5b,
	c=-2b
\end{eqnarray}$$
	
taking $$ b=1 $$, gives $$ a=-1 $$, $$ c=-2 $$, $$ d=5 $$ and the dimensionless parameter is:

\$$
	const=\frac{r^5\rho}{t^2E}
$$

Estimating the constant to be of order unity and using a photo such as this:
<br>
{::options parse_block_html="false" /}
<div style="text-align:center">
<img src='https://upload.wikimedia.org/wikipedia/commons/7/78/Trinity_Test_Fireball_16ms.jpg' width='300px'/> 
<br>
source: wikipedia.
</div>
<br>
taken at 25ms after the explosion with a blast radius of ~140m, he was able to get an estimation of 20kT which is the accurate value.
So both Fermi and Taylor used estimation methods and got an order of magnitude estimation which was very close to the exact
value, a feat which otherwise would require much more time and effort.