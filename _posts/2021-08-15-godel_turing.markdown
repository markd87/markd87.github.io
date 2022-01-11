---
layout: post
title: "Gödel & Turing"
date: 2021-08-15 16:43:00 +0000
tags: computation math philosophy
---

I've recently read Scott Aaronson's book "Quantum computing since Democritus", which covers in a very unique way computational complexity and quantum computing. One of my favorite parts was the connection drawn between Gödel's incompleteness theorem and Turing's Halting Problem. Two seminal theorems in logic and computer science. I wanted to reproduce the arguments here.

## Turing's Halting Problem (THP)

The halting problem is the statement that: given a program, there is no program that can determine if the given program will halt or not, without actually running it and waiting potentially an infinite amount of time.

The proof is by contradiction. Assume there exists a program P that solves the halting problem. We modify P to create a new program P', such that given another program Q as input, P' does the following:
(1) runs forever if Q halts given its code as input,
(2) halts if Q runs forever given its code as input .   

(We know if Q halts or not because we have P).

Then we input P' its own code as input and we get that P' will run forever if it halts or run forever if it halts, leading to a contradiction.


## Gödel's incompleteness theorems (GIT)

The incompleteness theorem is a profound statement about the true statement that a given set of axioms can prove.
Specifically, given any set of consistent and computable (i.e. finite or can be produced with an algorithm) axioms, there exists a true statement about the integers that can't be proven from these axioms.

Godel's second incompleteness theorem deals with the consistency of a system of axioms and their corresponding proofs. Specifically, it says that if a system of proofs ${\rm F}$ is consistent, then it cannot prove its own consistency. 
(Note that if it's not consistent then it can prove whatever it wants, since it's anyway inconsistent).


## Proof

At first there doesn't seem to be any connection between GIT and THP.
The proof follows by contradiction. Assume the incompleteness theorem was false,
which means there exists a set of axioms $F$ from which any statement about the integers can be proven true or false. Therefore, for any given program, we can go over all the (countable) proofs in $F$ and decide if the program halts or not. The crucial connection is that the decision if a program halts or not is a statement about the integers.
Since this contradicts the THP, we conclude that $F$ can't exist.
