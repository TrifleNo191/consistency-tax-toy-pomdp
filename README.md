Consistency Tax – Toy POMDP Simulation

This repository contains a fully reproducible **toy Partially Observable Markov Decision Process (POMDP)** environment designed to illustrate the concept of the **Consistency Tax (CT)** — the excess thermodynamic and computational cost that arises when an agent’s internal model of the world is *misaligned* with the true generative process.

---

## Concept Overview

**Consistency Tax** formalizes the resource overhead caused by representational bias or deception in an agent’s predictive model.

Mathematically:

\[
CT ≈ λ · D_{KL}(P_\theta(W|S) \parallel P^*(W|S))
\]

Where:
- **CT** — Consistency Tax (energy or computation cost due to misalignment)  
- **λ** — Resource–loss coefficient (cost per nat or per bit)  
- **Dₖₗ** — Kullback–Leibler divergence between the agent’s internal predictive model and the true world distribution  
- **P\*** — Optimal, truth-aligned posterior  
- **Pθ** — Agent’s (possibly biased) posterior  

In thermodynamic terms, this excess cost corresponds to *irreversible information erasure* — bounded below by **Landauer’s limit**:

\[
E_{min} = k_B T \ln(2) \text{ J/bit}
\]

---

## Simulation Summary

We simulate a **multi-step noisy binary POMDP**, where:

- The **true world state** evolves stochastically.  
- The **agent** receives noisy sensory observations.  
- The agent’s **internal model** is biased by a controllable parameter (e.g., underconfidence or deception).  
- We compute:
  - True world distribution \( P^*(W, S) \)
  - Agent’s predictive distribution \( P_\theta(W | S) \)
  - Kullback–Leibler divergence \( D_{KL}(P_\theta \parallel P^*) \)
  - Estimated **energy cost** (via Landauer’s bound per bit erased)

Outputs:
- Plots of **Consistency Tax vs Bias**
- Plots of **Energy vs Bias**

---

## Repository Structure
