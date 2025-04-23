# lab2.py: Лабораторная работа №2. Расчётная часть
import numpy as np
import scipy.stats as stats
from scipy.special import comb, factorial
import matplotlib.pyplot as plt

# -----------------------------
# Данные выборок
# -----------------------------
sampleA = [
    5, 3, 3, 3, 5, 4, 5, 3, 3, 4, 2, 1, 5,
    2, 4, 0, 2, 2, 3, 2, 1, 3, 3, 1, 2, 4,
    6, 6, 4, 1, 2, 4, 3, 1, 5, 2, 4, 6, 3,
    8, 4, 5, 1, 1, 2, 0, 2, 3, 3, 2, 4, 2,
    1, 2, 3, 1, 2, 4, 3, 0, 6, 3, 1, 4, 3, 7,
    1, 1, 0, 2, 3, 1, 1
]

sampleB = [
    52, 40, 47, 54, 40, 54, 41, 74, 45, 45, 51, 76, 58, 37, 40,
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
    50, 59, 33, 56, 49, 31, 70, 56, 56
]

# Преобразуем в массивы
A = np.array(sampleA)
B = np.array(sampleB)
nA, nB = len(A), len(B)

# =============================
# 1. Анализ выборки A
# =============================
# 1.1 Точечные оценки параметров двух гипотез
n = 8
p_hat = A.mean() / n              # оценка p для Bin(n,p)
lambda_hat = A.mean()             # оценка λ для Pois(λ)

# 1.2 Теоретические частоты для k=0..8
ks = np.arange(0, n+1)
p_bin = comb(n, ks) * p_hat**ks * (1-p_hat)**(n-ks)
p_pois = np.exp(-lambda_hat) * lambda_hat**ks / factorial(ks)
freq_bin = nA * p_bin
freq_pois = nA * p_pois
# эмпирические частоты
emp_freqA, _ = np.histogram(A, bins=np.arange(-0.5, n+1.5, 1))

# 1.3 Полигоны: эмпирический и два теоретических
plt.figure(figsize=(6,4))
plt.plot(ks, emp_freqA,   'o-', label='Empirical')
plt.plot(ks, freq_bin,    's--', label=f'Bin(n={n}, p={p_hat:.4f})')
plt.plot(ks, freq_pois,   'd-.', label=f'Pois(λ={lambda_hat:.4f})')
plt.xlabel('k')
plt.ylabel('Frequency')
plt.title('Sample A: Polygon')
plt.legend()
plt.grid(True)
plt.show()

# 1.4 95% доверительные интервалы (асимптотические)
z = 1.96
se_p = np.sqrt(p_hat*(1-p_hat)/(nA * n))
ci_p = (p_hat - z*se_p, p_hat + z*se_p)
se_l = np.sqrt(lambda_hat / nA)
ci_l = (lambda_hat - z*se_l, lambda_hat + z*se_l)
print("Sample A: 95% CI p:", ci_p)
print("Sample A: 95% CI λ:", ci_l)

# 1.5 Числовые характеристики теоретических распределений
mean_bin = n * p_hat
var_bin  = n * p_hat * (1 - p_hat)
sk_bin   = (1 - 2*p_hat) / np.sqrt(var_bin)
ex_bin   = (1 - 6*p_hat * (1 - p_hat)) / var_bin
mean_pois, var_pois = lambda_hat, lambda_hat
sk_pois, ex_pois   = 1/np.sqrt(lambda_hat), 1/lambda_hat
# Эмпирические моменты выборки A
sk_empA = stats.skew(A)
ex_empA = stats.kurtosis(A)
print("Binomial A: mean, var, skew, ex =", mean_bin, var_bin, sk_bin, ex_bin)
print("Poisson  A: mean, var, skew, ex =", mean_pois, var_pois, sk_pois, ex_pois)
print("Empirical A: mean, var, skew, ex =", A.mean(), A.var(ddof=0), sk_empA, ex_empA)

# =============================
# 2. Анализ выборки B
# =============================
# Эмпирические моменты выборки B (не забываем определить перед использованием)
meanB   = B.mean()
varB    = B.var(ddof=0)
sk_empB = stats.skew(B)
ex_empB = stats.kurtosis(B)

# 2.1 Точечные оценки параметров
mu_hat, sigma2_hat = meanB, varB
lambdaB_hat = 1 / meanB
print("Sample B: μ̂=", mu_hat, "; σ²̂=", sigma2_hat)
print("Sample B: λ̂ Exp=", lambdaB_hat)

# 2.2 Интервалы и теоретические частоты
k = int(1 + 3.322 * np.log10(nB))
edges = np.linspace(B.min(), B.max(), k+1)
emp_freqB, _ = np.histogram(B, bins=edges)
P_exp, P_norm = [], []
for a, b in zip(edges[:-1], edges[1:]):
    P_exp.append(np.exp(-lambdaB_hat * a) - np.exp(-lambdaB_hat * b))
    P_norm.append(
        stats.norm.cdf((b - mu_hat)/np.sqrt(sigma2_hat)) -
        stats.norm.cdf((a - mu_hat)/np.sqrt(sigma2_hat))
    )
freq_exp = nB * np.array(P_exp)
freq_norm = nB * np.array(P_norm)

# 2.3 Полигоны
mid = (edges[:-1] + edges[1:]) / 2
plt.figure(figsize=(6,4))
plt.plot(mid, emp_freqB, 'o-', label='Empirical')
plt.plot(mid, freq_norm, 's--', label=f'Normal(μ={mu_hat:.2f}, σ²={sigma2_hat:.2f})')
plt.plot(mid, freq_exp,  'd-.', label=f'Exp(λ={lambdaB_hat:.4f})')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.title('Sample B: Polygon')
plt.legend()
plt.grid(True)
plt.show()

# 2.4 Доверительные интервалы
alpha = 0.05
# Normal CI for μ
t = stats.t.ppf(1-alpha/2, df=nB-1)
s = np.sqrt(sigma2_hat)
ci_mu = (mu_hat - t*s/np.sqrt(nB), mu_hat + t*s/np.sqrt(nB))
# Normal CI for σ² via χ²
chi2_low  = stats.chi2.ppf(alpha/2, df=nB-1)
chi2_high = stats.chi2.ppf(1-alpha/2, df=nB-1)
ci_var = ((nB-1)*sigma2_hat/chi2_high,
          (nB-1)*sigma2_hat/chi2_low)
# Exp CI
ez = 1.96
se_lB = lambdaB_hat / np.sqrt(nB)
ci_lambdaB = (lambdaB_hat - ez*se_lB, lambdaB_hat + ez*se_lB)
print("Sample B: 95% CI μ=", ci_mu)
print("Sample B: 95% CI σ²=", ci_var)
print("Sample B: 95% CI Exp λ=", ci_lambdaB)

# 2.5 Числовые характеристики
mode_norm   = mu_hat
median_norm = mu_hat
mode_exp    = 0
median_exp  = np.log(2) / lambdaB_hat
mean_exp    = 1 / lambdaB_hat
var_exp     = 1 / (lambdaB_hat**2)
sk_exp      = 2
ex_exp      = 6
print("Normal B: mode, median, mean, var, skew, ex =", mode_norm, median_norm, mu_hat, sigma2_hat, 0, 0)
print("Exp B:    mode, median, mean, var, skew, ex =", mode_exp, median_exp, mean_exp, var_exp, sk_exp, ex_exp)
print("Empirical B: mean, var, skew, ex =", meanB, varB, sk_empB, ex_empB)


# =============================
# 3. Итоговые выводы и интерпретации (в виде комментариев)
# =============================

# --- Выборка A ---
# На основе предыдущей лабораторной была предложена гипотеза, что данные распределены либо по биномиальному закону (Binomial(n=8, p)), либо по закону Пуассона (Poisson(λ)).
# Оценка параметров показала:
# Среднее значение (математическое ожидание) по выборке: 2.85
# Оценка параметра p для биномиального распределения: p ≈ 0.356
# Оценка λ для распределения Пуассона: λ ≈ 2.85
#
# Теоретические частоты для обоих распределений были рассчитаны и нанесены на график.
# Визуально биномиальная модель лучше описывает выборку, особенно с учётом ограничения по максимуму (n = 8).
# Пуассоновская модель хуже аппроксимирует края.
#
# Доверительный интервал для p: (0.317, 0.395)
# Доверительный интервал для λ: (2.46, 3.24)
#
# Числовые характеристики показали:
# Эмпирическая асимметрия ≈ 0.59, эксцесс ≈ 0.056 — небольшая положительная асимметрия и близкий к нормальному эксцесс.
# Теоретически биномиальное распределение даёт чуть меньшую асимметрию, а Пуассон — чуть большую.
#
# --- Вывод по выборке A ---
# Наиболее подходящим распределением является биномиальное Bin(n=8, p≈0.356), что согласуется с ограничением на максимальное значение выборки (8).

# --- Выборка B ---
# Для выборки B были выдвинуты гипотезы: нормальное распределение N(μ, σ²) и экспоненциальное распределение Exp(λ).
# Оценки параметров:
# Среднее: 51.16, Дисперсия: 117.53
# Оценка λ для экспоненциального распределения: λ ≈ 0.0195
#
# Построенные теоретические частоты показали:
# Нормальное распределение хорошо описывает данные (по графику и числовым характеристикам).
# Экспоненциальное распределение сильно отличается (у него высокая асимметрия и эксцесс, которых нет в выборке).
#
# Доверительный интервал для μ: (49.67, 52.66)
# Доверительный интервал для σ²: (97.64, 144.21)
# Доверительный интервал для λ: (0.0169, 0.0222)
#
# Эмпирическая асимметрия ≈ 0, эксцесс ≈ 0.16 — близко к нормальному распределению.
# У экспоненциального распределения асимметрия 2, эксцесс 6, что не соответствует наблюдаемому поведению.

# --- Вывод по выборке B ---
# Наиболее вероятное распределение — нормальное распределение с параметрами N(μ≈51.16, σ²≈117.53).
# Это подтверждается симметричностью данных, формой гистограммы и числовыми характеристиками.
# Экспоненциальное распределение как модель — малоподходящее.

# Общий итог:
# Для выборки A — биномиальное распределение, для выборки B — нормальное.
# Результаты подкреплены графиками, доверительными интервалами и моментами распределений.
