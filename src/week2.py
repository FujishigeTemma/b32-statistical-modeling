import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

wing_lengths = np.genfromtxt("data/wing_length.csv", delimiter=",")
wing_lengths = wing_lengths[np.isfinite(wing_lengths)]

plt.figure(figsize=(8, 6))
plt.hist(wing_lengths, bins=20, density=True, alpha=0.7, color="blue")
plt.xlabel("Wing Length")
plt.ylabel("Density")
plt.title("Distribution of Housefly Wing Lengths")
plt.grid(True)
plt.savefig("outputs/distribution_plot.png")

mu, std = stats.norm.fit(wing_lengths)
print(f"Fitted Parameters: mu = {mu:.2f}, std = {std:.2f}")

x = np.linspace(min(wing_lengths), max(wing_lengths), 100)
pdf = stats.norm.pdf(x, mu, std)

plt.figure(figsize=(8, 6))
plt.hist(wing_lengths, bins=20, density=True, alpha=0.7, color="blue", label="Data")
plt.plot(x, pdf, "r-", linewidth=2, label="Fitted Distribution")
plt.xlabel("Wing Length")
plt.ylabel("Density")
plt.title(
    f"Fitted Distribution of Housefly Wing Lengths\nMean = {mu:.2f}, Std Dev = {std:.2f}"
)
plt.legend()
plt.grid(True)
plt.savefig("outputs/fitted_distribution_plot.png")
