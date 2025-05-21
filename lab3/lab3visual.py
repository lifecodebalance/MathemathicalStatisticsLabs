import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm


sampleA = np.array([
    5, 3, 3, 3, 5, 4, 5, 3, 3, 4, 2, 1, 5,
    2, 4, 0, 2, 2, 3, 2, 1, 3, 3, 1, 2, 4,
    6, 6, 4, 1, 2, 4, 3, 1, 5, 2, 4, 6, 3,
    8, 4, 5, 1, 1, 2, 0, 2, 3, 3, 2, 4, 2,
    1, 2, 3, 1, 2, 4, 3, 0, 6, 3, 1, 4, 3, 7,
    1, 1, 0, 2, 3, 1, 1])
sampleB = np.array([52, 40, 47, 54, 40, 54, 41, 74, 45, 45, 51, 76, 58, 37, 40,
    42, 53, 54, 65, 46, 65, 61, 55, 38, 66, 42, 56, 54, 40, 60,
    43, 49, 77, 64, 53, 64, 58, 54, 56, 53, 43, 35, 56, 34, 59,
    58, 66, 49, 49, 57, 48, 42, 46, 52, 59, 50, 62, 50, 55, 55,
    46, 53, 51, 50, 60, 30, 48, 56, 29, 74, 52, 60, 44, 62, 23,
    54, 40, 33, 20, 55, 42, 61, 54, 41, 45, 75, 59, 41, 51, 45,
    54, 52, 62, 69, 65, 49, 48, 63, 52, 46, 44, 55, 60, 54, 39,
    82, 67, 68, 34, 56, 51, 56, 48, 53, 47, 59, 51, 59, 66, 48,
    61, 42, 54, 33, 39, 47, 46, 47, 73, 63, 34, 44, 51, 46, 40,
    43, 30, 60, 61, 53, 47, 42, 56, 70, 48, 45, 65, 48, 48, 51,
    40, 57, 56, 33, 44, 43, 45, 35, 35, 56, 59, 66, 56, 52, 44,
    53, 49, 55, 25, 53, 48, 73, 38, 58, 72, 57, 46, 54, 55, 59,
    38, 53, 48, 68, 36, 53, 41, 55, 51, 50, 45, 50, 29, 60, 39,
    50, 59, 33, 56, 49, 31, 70, 56, 56])

# ----- График для выборки A -----
def plot_sampleA_vs_binomial(sampleA, n=8, p_hat=None):
    data = np.array(sampleA)
    nA = len(data)
    if p_hat is None:
        p_hat = np.mean(data) / n
    # Эмпирическая гистограмма
    values, counts = np.unique(data, return_counts=True)
    rel_freq = counts / nA
    # Теоретические частоты
    ks = np.arange(0, n + 1)
    theor_probs = binom.pmf(ks, n, p_hat)

    plt.figure(figsize=(6, 4))
    plt.bar(values, rel_freq, width=0.5, label="Эмпирическая частота", alpha=0.7)
    plt.plot(ks, theor_probs, 'ro-', label=f"Bin(n={n}, p={p_hat:.2f})")
    plt.xlabel("Значение")
    plt.ylabel("Относительная частота")
    plt.title("Сравнение с биномиальным распределением (Sample A)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ----- График для выборки B -----
def plot_sampleB_vs_normal(sampleB):
    data = np.sort(np.array(sampleB))
    nB = len(data)
    x = np.linspace(min(data), max(data), 500)
    mu, sigma = np.mean(data), np.std(data, ddof=0)

    # ЭФР (ступенчатая)
    y_emp = np.arange(1, nB + 1) / nB
    # Теоретическая CDF нормального распределения
    y_theor = norm.cdf(x, loc=mu, scale=sigma)

    plt.figure(figsize=(6, 4))
    plt.step(data, y_emp, where='post', label="Эмпирическая функция (ЭФР)")
    plt.plot(x, y_theor, 'r--', label=f"Normal CDF (μ={mu:.2f}, σ={sigma:.2f})")
    plt.xlabel("X")
    plt.ylabel("F(X)")
    plt.title("Сравнение ЭФР и нормального распределения (Sample B)")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_sampleA_vs_binomial(sampleA)
plot_sampleB_vs_normal(sampleB)
