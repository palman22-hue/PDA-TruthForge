import numpy as np
import matplotlib.pyplot as plt

# Simplified 32D PDA simulation
dim = 32
karma_idx = 30   # Heavily weighted meta dims
order_idx = 31
weights = np.ones(dim)
weights[[karma_idx, order_idx]] = 5.0  # 5x weighting

# Initial state
state_noncoerce = np.random.uniform(-0.5, 0.5, dim)
state_coerce = state_noncoerce.copy()

# Natural trajectory: toward positive emotions, order, karma
N = np.random.uniform(0.2, 0.8, dim) * weights
N /= np.linalg.norm(N)  # Unit direction

steps = 50
entropy_non = []
entropy_co = []
cum_entropy_non = 0
cum_entropy_co = 0
karma_non = 0
karma_co = 0
coherence_non = 1.0
coherence_co = 1.0

for _ in range(steps):
    # Non-coercive: F close to N
    F_non = N * np.random.uniform(0.8, 1.2)
    dev_non = F_non - N
    delta_fric_non = np.dot(dev_non, dev_non)
    delta_res_non = abs(karma_non) * 0.05  # Low resistance
    delta_coh_non = (1 - coherence_non) * 0.1
    delta_total_non = delta_fric_non + delta_res_non + delta_coh_non
    entropy_non.append(delta_total_non)
    cum_entropy_non += delta_total_non
    state_noncoerce += F_non * 0.02
    karma_non += 0.4  # Credit
    coherence_non = min(1.0, coherence_non + 0.01)
    
    # Coercive: F deviates strongly
    F_co = -N * np.random.uniform(0.5, 2.0) + np.random.uniform(-1, 1, dim)
    dev_co = F_co - N
    delta_fric_co = np.dot(dev_co, dev_co)
    delta_res_co = abs(karma_co) * 0.5  # High resistance from debt
    delta_coh_co = (1 - coherence_co) * 2.0
    delta_total_co = delta_fric_co + delta_res_co + delta_coh_co
    entropy_co.append(delta_total_co)
    cum_entropy_co += delta_total_co
    state_coerce += F_co * 0.02
    karma_co -= 0.8  # Debt
    coherence_co = max(0.0, coherence_co - 0.03)

print(f"Cumulative entropy - Non-coercive: {cum_entropy_non:.2f}, Coercive: {cum_entropy_co:.2f}")
print(f"Final karma - Non: {karma_non:.1f}, Co: {karma_co:.1f}")
print(f"Final coherence - Non: {coherence_non:.2f}, Co: {coherence_co:.2f}")

# Plot
plt.plot(entropy_non, label='Non-coercive (Ethical)')
plt.plot(entropy_co, label='Coercive (Forced)')
plt.xlabel('Time Steps')
plt.ylabel('Instantaneous ΔS_total')
plt.title('PDA Entropy Simulation: Coercion vs Non-Coercion')
plt.legend()
plt.show()