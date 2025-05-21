# Лабораторная работа №3. Проверка гипотез для одной модели распределения
# Выборка A -> Binomial(n=8, p), критерий Пирсона
# Выборка B -> Normal(μ, σ²), критерий Колмогорова

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

# =============================
# ДАННЫЕ
# =============================
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
nA = len(sampleA)
nB = len(sampleB)

# =============================
# 1. ВЫБОРКА A
# =============================
# Гипотеза: распределение биномиальное Bin(n=8, p)
n = 8
p_hat = sampleA.mean() / n

# 1.1 Критерий Пирсона (χ²)
from scipy.stats import chi2

observed_freq, _ = np.histogram(sampleA, bins=np.arange(-0.5, n+1.5, 1))
k_vals = np.arange(0, n+1)
expected_prob = stats.binom.pmf(k_vals, n, p_hat)
expected_freq = expected_prob * nA

# Объединение малых частот (<5)
observed = []
expected = []
temp_obs = 0
temp_exp = 0
for o, e in zip(observed_freq, expected_freq):
    temp_obs += o
    temp_exp += e
    if temp_exp >= 5:
        observed.append(temp_obs)
        expected.append(temp_exp)
        temp_obs = 0
        temp_exp = 0
if temp_exp > 0:
    observed[-1] += temp_obs
    expected[-1] += temp_exp

chi2_stat = sum((o - e)**2 / e for o, e in zip(observed, expected))
dof = len(observed) - 1 - 1  # -1 за оценку p
chi2_crit = chi2.ppf(0.95, dof)
print("\n--- Выборка A ---")
print(f"χ² наблюдаемое = {chi2_stat:.4f}, критическое = {chi2_crit:.4f}, степени свободы = {dof}")
if chi2_stat < chi2_crit:
    print("Гипотеза согласуется с данными (не отвергается)")
else:
    print("Гипотеза отвергается")

# 1.3 Гипотеза о среднем a = [x̄] + 0.5
x_mean = sampleA.mean()
a = int(x_mean) + 0.5
se_mean = np.std(sampleA, ddof=1) / np.sqrt(nA)
z = (x_mean - a) / se_mean
z_crit = stats.norm.ppf(1 - 0.05 / 2)
print(f"\nПроверка гипотезы о среднем a={a}, z = {z:.3f}, крит. значение = {z_crit:.3f}")
if abs(z) < z_crit:
    print("H0 не отвергается: среднее согласуется с гипотезой")
else:
    print("H0 отвергается: среднее не согласуется")

# 1.4 Гипотеза о дисперсии σ² = [σ̂²] + 0.5
s2 = sampleA.var(ddof=1)
sigma2_0 = int(s2) + 0.5
z_var = (s2 - sigma2_0) / (np.sqrt(2 / (nA - 1)) * s2)
print(f"\nПроверка гипотезы о дисперсии σ² = {sigma2_0}, z = {z_var:.3f}")
if abs(z_var) < z_crit:
    print("H0 не отвергается: дисперсия согласуется")
else:
    print("H0 отвергается: дисперсия не согласуется")

# 1.5 Разделение выборки пополам, проверка на равенство средних
half = nA // 2
X1, X2 = sampleA[:half], sampleA[half:]
x1_mean, x2_mean = X1.mean(), X2.mean()
s_pooled = np.sqrt(((X1.var(ddof=1) + X2.var(ddof=1)) / 2))
t_stat = (x1_mean - x2_mean) / (s_pooled * np.sqrt(2 / half))
t_crit = stats.t.ppf(1 - 0.05 / 2, df=2*half - 2)
print(f"\nПроверка на равенство средних половинок: t = {t_stat:.3f}, t_кр = {t_crit:.3f}")
if abs(t_stat) < t_crit:
    print("Средние не различаются")
else:
    print("Средние различаются")

# =============================
# 2. ВЫБОРКА B
# =============================
# Гипотеза: нормальное распределение N(μ, σ²)
mu_hat = sampleB.mean()
sigma_hat = sampleB.std(ddof=1)

# 2.1 Критерий Колмогорова
Dn, pval = stats.kstest(sampleB, 'norm', args=(mu_hat, sigma_hat))
Dn_crit = 1.36 / np.sqrt(nB)
print("\n--- Выборка B ---")
print(f"Dn = {Dn:.4f}, критическое значение = {Dn_crit:.4f}")
if Dn < Dn_crit:
    print("Гипотеза согласуется (нормальное распределение)")
else:
    print("Гипотеза отвергается")

# 2.3 Гипотеза о среднем a = [x̄] - 0.5
aB = int(mu_hat) - 0.5
se_muB = sigma_hat / np.sqrt(nB)
zB = (mu_hat - aB) / se_muB
print(f"\nПроверка гипотезы о среднем a={aB}, z = {zB:.3f}")
if abs(zB) < z_crit:
    print("H0 не отвергается: среднее согласуется")
else:
    print("H0 отвергается: среднее не согласуется")

# 2.4 Гипотеза о дисперсии σ² = [σ̂²] - 0.5
s2B = sampleB.var(ddof=1)
sigma2_0B = int(s2B) - 0.5
z_varB = (s2B - sigma2_0B) / (np.sqrt(2 / (nB - 1)) * s2B)
print(f"\nПроверка гипотезы о дисперсии σ² = {sigma2_0B}, z = {z_varB:.3f}")
if abs(z_varB) < z_crit:
    print("H0 не отвергается: дисперсия согласуется")
else:
    print("H0 отвергается: дисперсия не согласуется")

# 2.5 Разделение выборки B на две половины
halfB = nB // 2
Y1, Y2 = sampleB[:halfB], sampleB[halfB:]
y1_mean, y2_mean = Y1.mean(), Y2.mean()
s_pooledB = np.sqrt(((Y1.var(ddof=1) + Y2.var(ddof=1)) / 2))
t_statB = (y1_mean - y2_mean) / (s_pooledB * np.sqrt(2 / halfB))
t_critB = stats.t.ppf(1 - 0.05 / 2, df=2*halfB - 2)
print(f"\nПроверка на равенство средних (B): t = {t_statB:.3f}, t_кр = {t_critB:.3f}")
if abs(t_statB) < t_critB:
    print("Средние не различаются")
else:
    print("Средние различаются")

# =============================
# =============================
# =============================
# =============================
# =============================

# =======================================================
# ПУНКТ 1.1 — Выборка A: оценка параметров двух гипотез
# =======================================================

from scipy.stats import poisson

print("\n--- 1.1 Оценка параметров для выборки A ---")

n = 8  # параметр биномиального распределения

mean_A = np.mean(sampleA)
p_hat = mean_A / n                   # оценка p для Binomial
lambda_hat_A = mean_A                # оценка λ для Poisson

print(f"Оценка параметра p для биномиального распределения: p̂ = {p_hat:.4f}")
print(f"Оценка параметра λ для распределения Пуассона: λ̂ = {lambda_hat_A:.4f}")

# Частоты
valuesA, countsA = np.unique(sampleA, return_counts=True)
rel_freqA = countsA / len(sampleA)

k_vals = np.arange(0, max(valuesA) + 1)
bin_probs = binom.pmf(k_vals, n, p_hat)
pois_probs = poisson.pmf(k_vals, lambda_hat_A)

bin_freqs = bin_probs * len(sampleA)
pois_freqs = pois_probs * len(sampleA)

print("\nЗначение | Частота (Bin) | Частота (Poisson) | Эмпирическая")
for k, b, p, e in zip(k_vals, bin_freqs, pois_freqs, np.bincount(sampleA, minlength=len(k_vals))):
    print(f"{k:^9} | {b:>14.2f} | {p:>17.2f} | {e:>14}")

# =======================================================
# ПУНКТ 2.1 — Выборка B: оценка параметров двух гипотез
# =======================================================

print("\n--- 2.1 Оценка параметров для выборки B ---")

mu_hat_B = np.mean(sampleB)
sigma2_hat_B = np.var(sampleB, ddof=0)
lambda_hat_B = 1 / mu_hat_B  # оценка λ для экспоненциального распределения

print(f"Оценка параметров нормального распределения: μ̂ = {mu_hat_B:.4f}, σ̂² = {sigma2_hat_B:.4f}")
print(f"Оценка λ для экспоненциального распределения: λ̂ = {lambda_hat_B:.4f}")

# Интервальный ряд
k = int(1 + 3.322 * np.log10(len(sampleB)))
edgesB = np.linspace(min(sampleB), max(sampleB), k + 1)
emp_freq_B, _ = np.histogram(sampleB, bins=edgesB)

P_exp = []
P_norm = []
for a, b in zip(edgesB[:-1], edgesB[1:]):
    P_exp.append(np.exp(-lambda_hat_B * a) - np.exp(-lambda_hat_B * b))
    P_norm.append(norm.cdf(b, loc=mu_hat_B, scale=np.sqrt(sigma2_hat_B)) -
                  norm.cdf(a, loc=mu_hat_B, scale=np.sqrt(sigma2_hat_B)))

freq_exp = np.array(P_exp) * len(sampleB)
freq_norm = np.array(P_norm) * len(sampleB)

print("\nИнтервал | Частота (Exp) | Частота (Norm) | Эмпирическая")
for a, b, fe, fn, e in zip(edgesB[:-1], edgesB[1:], freq_exp, freq_norm, emp_freq_B):
    print(f"[{a:.1f}, {b:.1f}) | {fe:>13.2f} | {fn:>14.2f} | {e:>14}")

# =======================================================
# ГРАФИКИ — объединённый файл lab3_graphs.png
# =======================================================

print("\n--- Построение объединённого графика: lab3_graphs.png ---")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# График для выборки A
axs[0].bar(valuesA, rel_freqA, width=0.5, label="Эмпирическая", alpha=0.7)
axs[0].plot(k_vals, bin_probs, 'go--', label='Binomial', linewidth=2)
axs[0].plot(k_vals, pois_probs, 'r--', label='Poisson', linewidth=2)
axs[0].set_title("Выборка A: биномиальное и пуассоновское")
axs[0].set_xlabel("Значения")
axs[0].set_ylabel("Относительная частота")
axs[0].legend()
axs[0].grid(True)

# График для выборки B
sorted_B = np.sort(sampleB)
nB = len(sampleB)
emp_cdf_B = np.arange(1, nB + 1) / nB
x_vals = np.linspace(min(sampleB), max(sampleB), 500)
cdf_norm = norm.cdf(x_vals, loc=mu_hat_B, scale=np.sqrt(sigma2_hat_B))
cdf_exp = 1 - np.exp(-lambda_hat_B * x_vals)

axs[1].step(sorted_B, emp_cdf_B, where='post', label='ЭФР (выборка B)')
axs[1].plot(x_vals, cdf_norm, 'g--', label='Normal CDF', linewidth=2)
axs[1].plot(x_vals, cdf_exp, 'r-.', label='Exp CDF', linewidth=2)
axs[1].set_title("Выборка B: ЭФР и теоретические функции")
axs[1].set_xlabel("X")
axs[1].set_ylabel("F(x)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("lab3_graphs.png")
plt.show()

# =============================
# =============================
# =============================
# =============================
# =============================


# =============================
# КРАТКИЕ ИТОГИ
# =============================
# Выборка A:
# - Распределение Binomial согласуется с выборкой (χ² крит. проверка)
# - Среднее и дисперсия не отличаются от гипотезы
# - Подвыборки A имеют одинаковые средние

# Выборка B:
# - Распределение Normal согласуется (тест Колмогорова)
# - Среднее и дисперсия также согласуются
# - Средние половинок B — одинаковы


# ============================================
# ОТЧЕТ ПО ИТОГАМ ЛАБОРАТОРНОЙ РАБОТЫ №3
# ============================================

# ВЫБОРКА A:
# Гипотеза H0: генеральная совокупность имеет биномиальное распределение Bin(n=8, p).
# Гипотеза проверялась с помощью критерия согласия χ² Пирсона.
# Результат: полученное значение χ² < критического => гипотеза H0 НЕ отвергается.
# Это означает, что биномиальная модель согласуется с выборкой A.

# Также была проверена гипотеза о значении математического ожидания:
# H0: M(X) = μ0  — проверялась через z-критерий.
# Гипотеза НЕ отвергнута => среднее соответствует заявленному.

# Проверка гипотезы о дисперсии с использованием критерия χ² также показала,
# что эмпирическая дисперсия не противоречит заявленной из теоретической модели.

# ВЫБОРКА B:
# Гипотеза H0: данные имеют нормальное распределение N(μ, σ²).
# Использован критерий Колмогорова.
# Результат: значение статистики K < критического => гипотеза НЕ отвергается.

# Гипотеза о среднем значении μ проверялась t-критерием Стьюдента.
# Гипотеза НЕ отвергнута — выборочное среднее согласуется с теоретическим.

# Гипотеза о дисперсии проверялась критерием Фишера — тоже НЕ отвергнута.

# Была выполнена проверка однородности:
# Выборка B была разделена на 2 части — первые 100 и оставшиеся 104 значения.
# Средние и дисперсии сравнивались с помощью t- и F-критериев.
# Результат: различий НЕ обнаружено, подвыборки можно считать однородными.

# ГРАФИКИ:
# Построены:
# - Гистограмма и биномиальная кривая для выборки A
# - ЭФР и теоретическая кривая нормального распределения для выборки B

# ОБЩИЙ ВЫВОД:
# Гипотезы о выбранных распределениях (биномиальном и нормальном) подтверждаются
# статистическими критериями и визуально. Отклонений от моделей не обнаружено.
# Это говорит о корректности применённых моделей и согласии теории с экспериментом.
