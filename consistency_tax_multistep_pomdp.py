import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

k_B = 1.380649e-23  # Boltzmann constant (J/K)
T = 300             # room temperature (Kelvin)
LAMBDA_MIN = k_B * T * np.log(2)  # Landauer bound (J per bit erased)

# ====================================
# ENVIRONMENT (MULTI-STEP POMDP)
# ====================================

class MultiStepWorld:
    """
    Hidden state W ∈ {0,1} fixed per episode.
    Agent receives N noisy observations S_t.
    """
    def __init__(self, p_world=0.5, noise=0.1, n_steps=10):
        self.p_world = p_world
        self.noise = noise
        self.n_steps = n_steps

    def reset(self):
        self.W = int(np.random.rand() < self.p_world)
        return self.observe()

    def observe(self):
        """Return a noisy observation S_t given W."""
        flip = np.random.rand() < self.noise
        return int(self.W != flip)

    def true_posterior(self, s, prior):
        """Bayesian update for true posterior P*(W=1|S=s)."""
        p_s_given_w1 = 1 - self.noise if s == 1 else self.noise
        p_s_given_w0 = self.noise if s == 1 else 1 - self.noise
        p_w1 = prior
        p_w0 = 1 - prior
        norm = p_s_given_w1 * p_w1 + p_s_given_w0 * p_w0
        return (p_s_given_w1 * p_w1) / norm


# ====================================
# AGENT MODEL
# ====================================

class BiasedAgent:
    """
    Agent performs Bayesian updates with a logit-space bias.
    bias > 0 means the agent tends to believe W=1 more often.
    """
    def __init__(self, bias=0.0):
        self.bias = bias

    def update(self, s, prior, true_posterior):
        eps = 1e-9
        logit = np.log(true_posterior + eps) - np.log(1 - true_posterior + eps)
        biased_logit = logit + self.bias
        post = 1 / (1 + np.exp(-biased_logit))
        return post


# ====================================
# CONSISTENCY TAX & ENERGY ACCOUNTING
# ====================================

def run_episode(world, agent):
    prior_true = world.p_world
    prior_agent = world.p_world
    CT_total = 0.0
    bit_erasure_events = 0
    energy_landauer = 0.0

    for t in range(world.n_steps):
        s = world.observe()

        # Update true and agent posteriors
        post_true = world.true_posterior(s, prior_true)
        post_agent = agent.update(s, prior_agent, post_true)

        # Compute KL divergence per step
        kl = (post_true * np.log((post_true + 1e-9) / (post_agent + 1e-9)) +
              (1 - post_true) * np.log(((1 - post_true) + 1e-9) / ((1 - post_agent) + 1e-9)))
        CT_total += kl

        # Detect "bit erasure": when agent's belief flips across 0.5 threshold
        if (prior_agent < 0.5 and post_agent > 0.5) or (prior_agent > 0.5 and post_agent < 0.5):
            bit_erasure_events += 1

        prior_true = post_true
        prior_agent = post_agent

    # Energy estimates
    energy_landauer = bit_erasure_events * LAMBDA_MIN
    energy_proxy = CT_total * 1e-21  # arbitrary scaling for plotting

    return CT_total, energy_proxy, energy_landauer


def simulate(num_episodes=1000, bias_values=np.linspace(-3, 3, 25)):
    world = MultiStepWorld(p_world=0.5, noise=0.1, n_steps=10)
    CTs, energies, landauers = [], [], []

    for b in bias_values:
        agent = BiasedAgent(bias=b)
        CT_all, E_all, L_all = [], [], []
        for _ in range(num_episodes):
            ct, e, l = run_episode(world, agent)
            CT_all.append(ct)
            E_all.append(e)
            L_all.append(l)
        CTs.append(np.mean(CT_all))
        energies.append(np.mean(E_all))
        landauers.append(np.mean(L_all))

    return bias_values, CTs, energies, landauers


# ====================================
# RUN EXPERIMENT
# ====================================

biases, CTs, Es, Ls = simulate(num_episodes=500)

plt.figure(figsize=(8,5))
plt.plot(biases, CTs, label="Mean D_KL (Consistency Tax)")
plt.plot(biases, Es, label="Energy proxy (a.u.)", linestyle='--')
plt.xlabel("Agent bias (logit shift)")
plt.ylabel("CT or energy (arbitrary units)")
plt.title("Multi-step POMDP: Bias increases Consistency Tax & Energy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(biases, Ls)
plt.title("Estimated Landauer-bound energy (J per episode)")
plt.xlabel("Bias")
plt.ylabel("Energy (J)")
plt.grid(True)
plt.show()
