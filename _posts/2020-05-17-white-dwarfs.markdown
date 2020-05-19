---
layout: post
title:  "White Dwarfs"
date:   2020-05-17 20:00:00 +0100
categories: physics
---

A white dwarf is the amusing name given to the last stage of the life of medium sized stars (<5 solar masses). 
Stars are huge nuclear furnaces, where nuclear fusion (converting hydrogen nuclei to helium nuclei ) provides the energy, which is released as light and heat and provides sufficient pressure against the star's gravity. However, once the nuclear fusion fuel runs out, there is nothing else to hold the star. Or is there?

The first white dwarf was discovered in 1862, Sirius B, the companion star in the Sirius binary system. At the time it was just another star, much smaller and less bright than Sirius A (the brightest star in the night sky), however once the star's radius (0.008 $R_{sun}$) and mass ($~1M_{sun}$) were calculated (from its Luminosity using Stefan-Boltzman Law and from its orbit around Sirius A), it was evident that it is no usual star, it's the mass of the sun tightly packed in a volume the size of the earth.

The reason I've decided to write about white dwarfs is because it's one of the best examples of quantum mechanics manifesting itself on the largest scales, scales where typically only gravity is relevant.

From the observed luminosity one can obtain the temperature in the center of the star ($\sim 10^7 K$), this temperature shows the interior of the stars cannot be Hydrogen, as that together with the high density would result in nuclear fusion, resulting in far higher luminosity than observed. Indeed the interior are typically Carbon and Oxygen, the remnants of the nuclear fusion reactions in the star's main sequence life. Similarly, one can obtain the radiation pressure in the centre of the star assuming an ideal gas, showing it is insufficient to hold the star against gravity. So what holds white dwarfs?

This is where quantum mechanics comes in to play. The high density of white dwarfs, results in the electrons being sufficiently confined that quantum effects show up. The first quantum principle at play is that of Paul's exclusion principle, which says that two electrons cannot occupy the same state. For the electrons in the white dwarf, the states are defined by the electrons' momentum, position and spin, where each electron occupies a volume of $h^3$, with $h$ being the Planck constant, (The actual volume is $h^3/2$ due to the two spin states, allowing two electrons in a phase-space volume $h^3$). The electrons will occupy all available states up to the highest state with momentum $P_f$ (Fermi momentum). If the Fermi momentum or energy is high enough (relative to the thermal energy), it means that electrons from lower state cannot get excited to states above the Fermi state (due to Pauli's exclusion principle), resulting in what is called a degenerate electron gas. The degeneracy of the electrons, and the fact that the electrons cannot occupy any lower energy states results in a pressure opposing the gravitational collapse of the star.

This competition between classical gravity and the quantum Pauli's exclusion principle, will result in an equilibrium at a certain radius, which will determine the radius of the white dwarf. Here I'll obtain an expression for it using dimensional analysis.

Assuming the star contains $N$ atoms with nuclear number A of protons and neutrons, i.e. $N=M/(Am_p)$.

The number of electrons from each atom is $Z$ giving the total number of electrons: $N_e = \frac{M}{m_p}\frac{Z}{A}.

The energy of an electron in the gas is: $E\sim\frac{p^2}{m_e}$. From Heisenberg's uncertainty principle $\Delta x \Delta p \sim \hbar$, which means confining the electron in location $\Delta x ~ n_e^{-1/3}$, where $n_e \sim \frac{N_e}{R^3}$ is the number density of electrons, results in an increase in it's momentum, which is another way to understand the resulting pressure of the degenerate electron gas. Therefore, the electron kinetic energy of the electron gas can be written as 
\begin{equation}
E_{kin} \sim \frac{\hbar^2 N_e^{5/3}}{m_e R^2}.
\end{equation}
Inserting $N_e$,
\begin{equation}
E_{kin} \sim \frac{\hbar^2 M^{5/3}}{m_e m_p^{5/3} R^2}\left(\frac{Z}{A}\right)^{5/3}.
\end{equation}
 The gravitational energy is given by,
\begin{equation}
E_g = \frac{GM^2}{R}.
\end{equation}
Comparing the two energy scales (the idea of comparable energy scales in a system is also known as equi-partition) gives the equilibrium white dwarf radius as a function of its mass,
\begin{equation}
R\sim \left(\frac{\hbar c}{m c^2}\right)\left(\frac{\hbar c}{Gm_p^2}\right)\left(\frac{m_p}{M }\right)^{1/3}\left(\frac{Z}{A}\right)^{5/3}.
\end{equation}
Conveniently written in terms of dimensionless quantities, apart from the first term where the units are length. ($\hbar c $ has units of energy times length, personally I remember it as $197eV \cdot nm$ ).

