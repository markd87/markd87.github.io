---
layout: post
title: "White Dwarfs"
date: 2020-05-17 20:00:00 +0100
tags: physics
---

White Dwarfs - where relativity, quantum mechanics and gravity meet.

A white dwarf is the amusing name given to the last stage of the life of medium sized stars (<5 solar masses).
Stars are huge nuclear furnaces, where nuclear fusion (converting hydrogen nuclei to helium nuclei ) provides the energy, which is released as light and heat and provides sufficient pressure against the star's gravity. However, once the nuclear fusion fuel runs out, there is nothing else to hold the star. Or is there?

The first white dwarf was discovered in 1862, Sirius B, the companion star in the Sirius binary system. At the time it was just another star, much smaller and less bright than Sirius A (the brightest star in the night sky), however once the star's radius (0.008 $R_{sun}$) and mass ($\sim 1M_{sun}$) were calculated (from its Luminosity using Stefan-Boltzmann Law and from its orbit around Sirius A), it was evident that it is no usual star, it's the mass of the sun tightly packed in a volume the size of the earth.

The reason I've decided to write about white dwarfs is because it's one of the best examples of quantum mechanics manifesting itself on the largest scales, scales where typically only gravity is relevant.

From the observed luminosity one can obtain the temperature in the center of the star ($\sim 10^7 K$), this temperature shows the interior of the stars cannot be Hydrogen, as that together with the high density would result in nuclear fusion, resulting in far higher luminosity than observed. Indeed the interior are typically Carbon and Oxygen, the remnants of the nuclear fusion reactions in the star's main sequence life. Similarly, one can obtain the radiation pressure in the centre of the star assuming an ideal gas, showing it is insufficient to hold the star against gravity. So what holds white dwarfs?

This is where quantum mechanics comes in to play. The high density of white dwarfs, results in the electrons being sufficiently confined that quantum effects show up. The first quantum principle at play is that of Pauli's exclusion principle, which says that two electrons cannot occupy the same state. For the electrons in the white dwarf, the states are defined by the electron momentum, position and spin, where each electron occupies a volume of $h^3$ (volume in phase space momentum and position), with $h$ being Planck's constant (The actual volume is $h^3/2$ due to the two spin states, allowing two electrons in the same phase-space volume). The electrons will occupy all available states up to the highest state having momentum $P_f$ (Fermi momentum). If the Fermi momentum or energy is high enough (relative to the thermal energy), it means that electrons from lower states cannot get excited to states above the Fermi state (due to Pauli's exclusion principle), resulting in what is called a degenerate electron gas. The degeneracy of the electrons and the fact that the electrons cannot occupy any lower energy states results in a "quantum pressure" opposing the gravitational collapse of the star.

This competition between classical gravity and the quantum Pauli's exclusion principle, will result in an equilibrium at a certain radius, which will determine the radius of the white dwarf. Here I'll obtain an expression for it using dimensional analysis.

Assuming the star contains $N$ atoms with nuclear number A of protons and neutrons, i.e. $N=M/(Am_p)$, the number of electrons from each atom is $Z$ giving the total number of electrons: $N_e = \frac{M}{m_p}\frac{Z}{A}$.

The kinetic energy of an electron in the gas is: $E\sim\frac{p^2}{m_e}$. From Heisenberg's uncertainty principle $\Delta x \Delta p \sim \hbar$, which means confining the electron in location $\Delta x \sim n_e^{-1/3}$, where $n_e \sim \frac{N_e}{R^3}$ is the number density of electrons, results in an increase in its momentum, which is another way to understand the resulting pressure of the degenerate electron gas.

Therefore, the electron kinetic energy of the electron gas can be written as

$$
E_{kin} \sim \frac{\hbar^2 N_e^{5/3}}{m_e R^2}.
$$

Inserting $N_e$,

$$
E_{kin} \sim \frac{\hbar^2 M^{5/3}}{m_e m_p^{5/3} R^2}\left(\frac{Z}{A}\right)^{5/3}.
$$

The gravitational energy is given by,

$$
E_g = \frac{GM^2}{R}.
$$

Comparing the two energy scales (the idea of comparable energy scales in a system is also known as equi-partition) gives the equilibrium white dwarf radius as a function of its mass,

$$
R\sim \left(\frac{\hbar c}{m_e c^2}\right)\left(\frac{\hbar c}{Gm_p^2}\right)\left(\frac{m_p}{M }\right)^{1/3}\left(\frac{Z}{A}\right)^{5/3}.
$$

Conveniently written in terms of dimensionless quantities, apart from the first term where the units are length. ($\hbar c $ has units of energy times length, equal to $\sim 200 {\rm eV}\cdot{\rm nm} = 200\times 10^{-12} {\rm eV}\cdot {\rm km}$ ).

Further rearranging and expressing the star's mass in terms of the Sun's mass,

$$
R\sim \left(\frac{\hbar c}{m_e c^2}\right) \left( \frac{m_{Planck}}{m_p}\right)^2\left(\frac{Z}{A}\right)^{5/3}
\left(\frac{m_p}{M_{Sun}} \right)^{1/3}
\left(\frac{M}{M_{Sun}}\right)^{-1/3},
$$

where I also introduced Planck's mass, $m_{Planck} = \sqrt{\frac{\hbar c}{G}}$.
Mentioning Planck's name in any constant, implies Quantum effects are present.

Assuming Carbon for the interior of the star $Z=6, A=12$,

$$
R\ [{\rm km}]\sim 2000
\left(\frac{M}{M_{Sun}}\right)^{-1/3}
$$

(A reminder that the earth's radius is $\sim 6,400 \ {\rm km}$)
The surprising result being, the white dwarf's radius gets smaller as the mass increases. This suggests a limiting point, where something breaks down, and the radius cannot get any smaller while being sustained through the electron degeneracy pressure.

As the mass increases and radius decreases, the electron density increases and correspondingly the electron kinetic energy increases. At some point, the electron becomes relativistic, meaning that its kinetic energy becomes comparable or larger than its rest energy, at which point the electron is effectively massless, as its kinetic energy is described by a photon's energy-momentum relation, $E_{kin}=pc$. The relativistic electron gas is then,

$$
E_{kin} \sim \frac{\hbar c M^{4/3}}{m_p^{4/3} R}\left(\frac{Z}{A}\right)^{4/3}.
$$

Equating it to the gravitational energy gives,

$$
M \sim
\frac{m_{Planck}^3}{m_p^2}
\left(\frac{Z}{A}\right)^2.
$$

The mass is a constant independent of the radius.
This limit for the mass of a white dwarf is known as the Chandrasekhar limit, named after [Subrahmanyan Chandrasekhar](https://en.wikipedia.org/wiki/Subrahmanyan_Chandrasekhar) who derived this limit in the 1930's and received a Nobel prize for it. As the result is a constant, the pre-factors are important, and the exact value is $\sim 1.4 \ M_{sun}$.

### Summary

> When a star exhausts its fuel, the star contracts, increasing its density. The electrons get confined enough to become degenerate, at which point Pauli's Exclusion principle results in a pressure opposing gravity. When the mass is large enough, electrons become relativistic, resulting in a unique limiting mass.
