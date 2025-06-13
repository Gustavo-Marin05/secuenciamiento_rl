import pickle
import matplotlib.pyplot as plt

with open("rewards_history.pkl", "rb") as f:
    rewards = pickle.load(f)

plt.plot(range(1, len(rewards) + 1), rewards, marker='o')
plt.xlabel("Iteración")
plt.ylabel("Recompensa media")
plt.title("Evolución de recompensa media durante el entrenamiento")
plt.grid(True)
plt.show()
