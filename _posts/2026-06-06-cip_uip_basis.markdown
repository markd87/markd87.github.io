---
layout: post
title: "CIP, UIP and Basis"
date: 2026-06-06
tags: "finance"
---


An FX forward contract sets the price today at which an investor can buy or sell a currency at a given date in the future.

## CIP

CIP (covered interest parity) is based on the no arbitrage principle, which determines the forward price.

CIP is based on equating two approaches to satisfy no-arbitrage. Assume we start with EUR, we could either invest in a risk-free asset with rate $r_f$ and make $e^{r_f T}$ or we can convert it to USD using spot $S_t$, invest in USD risk-free asset with rate $r_d$ to make $S_t e^{r_d T}$ and at the same time sell a forward with strike $f_t$ to sell the USD back to EUR. The second approach then makes $\frac{S_t}{f_t} e^{r_d T}$. For the two approaches to be equivalent we need: $f = S_t e^{r_d-r_f}$. Selling the forward, is the hedging part which is why it's called covered.

A more formal deviation of the forward price: consider a strategy where at time $t$ an investor borrows $S_t$ USD to buy 1 EUR, and invest in a EUR denominated asset at rate $r_f$. At the same time the investor enters a forward contract to sell $e^{r_f\tau}$ EUR notional of a forward with maturity T, with strike $f_t(T)$. At time T, the 1 EUR, is now worth $e^{r_f\tau}$, where  $\tau = T-t$.T the forward contract settles and the investor sells the $e^{r_f\tau}$ EUR in exchange for $e^{r_f\tau}f_t(T)$ USD. The investor also repays the borrowed USD plus interest: $S_t e^{r_d\tau}$ USD. Here, $r_d, r_f$ are the domestic and foreign interest rates, with the convention EUR/USD written as foreign/domestic, i.e. how much USD per 1 EUR.
The payoff of this zero-cost strategy must be 0 from the no-arbitrage condition, i.e. $e^{r_f\tau}f_t(T) - S_te^{r_d\tau} = 0$. Rearranging gives: 

$f_t(T) = S_t e^{(r_d - r_f)\tau}$.

The forward is determined by the spot and the interest rate differential. The higher the differential the higher the forward rate relative to spot. 

## Cross Currency Basis

The forward rate in the markets doesn't actually satisfy CIP since the GCF (2008 crisis), the observed forward in the market is:

$f_t^m(T) = e^{(r_d-(r_f + \epsilon_B))\tau}$,

where $\epsilon_B$ is the basis, which represent the extra demand for the domestic currency. For USD as the domestic currency, a demand for USD funding, or USD funding shortages implies a negative basis. A negative basis means that the forward is higher than implied by CIP, i.e. a more unfavorable rate for buying EUR and selling USD in the future, following selling EUR and obtaining USD at the present.
If many people want USD now (high USD funding demand), they are willing to accept a worse future exchange rate on that forward leg.

## UIP

UIP is uncovered interest rate parity. It's the expectation of the future spot being equal to the forward:

$ E_t[S_T] = f_t(T) \approx S_t ( 1 + (r_d-r_f) )$.

So if UIP holds, for example using USDJPY, with dom = JPY, for = USD, since r_f > r_d, the expectation is that USD will depreciate relative to the JPY, exactly by the rate differential. This means that if we borrow in JPY, buy USD and invest in asset with interest $r_f$, the gain from the interest will be offset by the depreciation of the currency.
In practice, UIP doesn't hold, and this is exactly the famous carry trade, so why is UIP violated in the markets? the reason is risk premium, i.e. compensation through higher expected returns for taking risk.