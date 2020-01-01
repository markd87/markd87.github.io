---
layout: post
title:  "Compound interest"
date:   2020-01-01 14:35:00 +0000
categories: posts
---

A compound interest is a fixed interest that is applied every given 
time period to an initial value (typically money). As the interest is 
reapplied after every time period we get an interest of an interest, or 
a compound interest. The total accumulated amount after 
$$n$$ time periods is given by,

\begin{equation}
\label{eq:ci}
A = P (1+\frac{r}{100})^n
\end{equation}

Where $$A$$ is the total accumulated amount, \\
$$P$$ the initial amount,\\
$$r$$ the interest rate, \\
and $$n$$ integer number of periods.
\\
\\
Exponential growth is not always easy to calculate in one's mind or appreciate
the change after a given time passes, as is linear change. Therefore simplification to the above formula can be handy to get a quick idea of the impact of a given interest rate on the final accumulated amount relative to the initial amount.
\\
\\
The underlying assumption behind the simplification is that $$\frac{r}{100} \ll 1$$, which is typically satisfied ($$r$$ is typically smaller than $$10$$), however we will see what are the corresponding corrections.

Taking the $$\ln$$ of both sides of $$(\ref{eq:ci})$$,

\begin{equation}
\label{eq:lci}
\ln(\frac{A}{P}) = n \ln(1+\frac{r}{100})
\end{equation}

Next, we use the Taylor expansion of $$\ln(1+x)$$ around $$x=0$$, given by:
$$\ln(1+x) = x-\frac{x^2}{2}+O(x^3)$$.
Using this in $$\ref{eq:lci}$$ gives,

\begin{equation}
\label{eq:simp_ci}
\ln(\frac{A}{P}) = n (\frac{r}{10^2}-\frac{r^2}{2\times 10^4})
\end{equation}
\\
\\
Assuming we want to know how many time periods $$n$$ we need to wait with compound interest $$r$$ for a given increase in the accumulated amount $$A$$ from an initial amount $$P$$, we solve for $$n$$,
\begin{equation}
\label{eq:solve_ci}
n = \frac{100\ln(\frac{A}{P})}{r}\frac{1}{1-\frac{r}{200}}
\end{equation}

We can further apply a Taylor expansion for $$\frac{1}{1-x}=1+x+O(x^2)$$,
giving,

\begin{equation}
\label{eq:further_ci}
n = \frac{100\ln(\frac{A}{P})}{r}(1+\frac{r}{200}+O(\frac{r^2}{4\times 10^4})).
\end{equation}

Finally, we can write

\begin{equation}
\label{eq:final_ci}
n = \frac{100\ln(\frac{A}{P})}{r}+\frac{\ln(\frac{A}{P})}{2}.
\end{equation}

As a useful example, for doubling of the initial amount, i.e. $$\frac{A}{P}=2$$, we get

\begin{equation}
\label{eq:rule_2}
n_2 \approx \frac{69.3}{r}+0.35.
\end{equation}

However, since 69.3 or 69 is not an easy number to divide, the closest number 
which has many divisors and is larger but closest to 69, is 72, which also allows to drop the additional constant value, giving

\begin{equation}
\label{eq:rule_72}
n_2 \approx \frac{72}{r},
\end{equation}

also known as the rule of 72.

For example, to double an initial investment with 6% yearly return, one would need
to wait, ~$$12$$ years.