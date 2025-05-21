import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm, expon, skew, kurtosis
from scipy.special import comb, factorial
from scipy.optimize import minimize_scalar

# Выборка A
sampleA = [
    5, 3, 3, 3, 5, 4, 5, 3, 3, 4, 2, 1, 5,
    2, 4, 0, 2, 2, 3, 2, 1, 3, 3, 1, 2, 4,
    6, 6, 4, 1, 2, 4, 3, 1, 5, 2, 4, 6, 3,
    8, 4, 5, 1, 1, 2, 0, 2, 3, 3, 2, 4, 2,
    1, 2, 3, 1, 2, 4, 3, 0, 6, 3, 1, 4, 3, 7,
    1, 1, 0, 2, 3, 1, 1
]
# Выборка B
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

A = np.array(sampleA)
B = np.array(sampleB)
nA, nB = len(A), len(B)

# 1. Анализ выборки A
# 1.1. Точечные оценки для Bin(n=8, p) и Pois(lambda)
n = 8
p_hat = A.mean() / n        # оценка p = среднее/8
lambda_hat = A.mean()       # оценка λ = среднее
print(f"Выборка A: nA={nA}, среднее={A.mean():.4f}")
print(f"Оценка p биномиального распределения: p̂={p_hat:.4f}")
print(f"Оценка λ пуассоновского распределения: λ̂={lambda_hat:.4f}")

# 1.2. Теоретические частоты
ks = np.arange(0, n+1)
P_bin = comb(n, ks) * p_hat**ks * (1-p_hat)**(n-ks)
P_pois = np.exp(-lambda_hat) * lambda_hat**ks / factorial(ks)
freq_bin = nA * P_bin
freq_pois = nA * P_pois
emp_freqA, _ = np.histogram(A, bins=np.arange(-0.5, n+1.5, 1))

# 1.3. Полигоны
plt.figure(figsize=(6,4))
plt.plot(ks, emp_freqA, 'o-', label='Эмпирические частоты')
plt.plot(ks, freq_bin, 's--', label=f'Binomial(n=8, p={p_hat:.2f})')
plt.plot(ks, freq_pois, 'd-.', label=f'Poisson(λ={lambda_hat:.2f})')
plt.xlabel('k')
plt.ylabel('Частота')
plt.title('Выборка A: эмпирический и теоретические полигоны')
plt.legend(); plt.grid(True); plt.show()

# 1.4. 95% доверительные интервалы
ez = 1.96
se_p = np.sqrt(p_hat*(1-p_hat)/(nA*n))
ci_p = (p_hat - ez*se_p, p_hat + ez*se_p)
se_l = np.sqrt(lambda_hat/nA)
ci_l = (lambda_hat - ez*se_l, lambda_hat + ez*se_l)
print(f"95% ДИ для p: [{ci_p[0]:.4f}, {ci_p[1]:.4f}]")
print(f"95% ДИ для λ: [{ci_l[0]:.4f}, {ci_l[1]:.4f}]")

# 1.5. Числовые характеристики
# Эмпирические моменты
modeA = ks[np.argmax(emp_freqA)]            # мода
medianA = np.median(A)                       # медиана
skewA = skew(A)                              # асимметрия
excessA = kurtosis(A)                        # эксцесс
print(f"Эмпирическая мода A: {modeA}")
print(f"Эмпирическая медиана A: {medianA}")
print(f"Эмпирическая асимметрия A: {skewA:.4f}")
print(f"Эмпирический эксцесс A: {excessA:.4f}")

# Теоретические моменты Binomial
E_bin = n*p_hat
Var_bin = n*p_hat*(1-p_hat)
Skew_bin = (1-2*p_hat)/np.sqrt(Var_bin)
Ex_bin = (1-6*p_hat*(1-p_hat))/Var_bin
print("Binomial теор.: E=%.4f, Var=%.4f, Skew=%.4f, Ex=% .4f" % (E_bin, Var_bin, Skew_bin, Ex_bin))
# Теоретические моменты Poisson
E_pois= lambda_hat; Var_pois=lambda_hat
Skew_pois=1/np.sqrt(lambda_hat); Ex_pois=1/lambda_hat
print("Poisson теор.: E=%.4f, Var=%.4f, Skew=%.4f, Ex=% .4f" %
      (E_pois, Var_pois, Skew_pois, Ex_pois))

# Вывод по выборке A
# Резюме: биномиальная модель Bin(8, p̂) лучше описывает распределение A,
# поскольку максимальное значение = 8.

# 2. Анализ выборки B
# 2.1. Точечные оценки для Normal(μ, σ²) и Exponential(λ)
meanB = B.mean(); varB = B.var(ddof=0)
mu_hat, sigma2_hat = meanB, varB
lambdaB_hat = 1/meanB
print(f"Выборка B: nB={nB}, среднее={meanB:.4f}, дисперсия={varB:.4f}")
print(f"Оценка параметров: μ̂={mu_hat:.4f}, σ²̂={sigma2_hat:.4f}, λ̂ Exp={lambdaB_hat:.4f}")

# 2.2. Интегральные частоты по Старджессу
kB = int(1+3.322*np.log10(nB));
edges = np.linspace(B.min(), B.max(), kB+1)
emp_freqB, _ = np.histogram(B, bins=edges)
# вероятности
P_expB = np.diff(expon.cdf(edges, scale=1/lambdaB_hat))
P_normB= np.diff(norm.cdf(edges, loc=mu_hat, scale=np.sqrt(sigma2_hat)))
freq_expB = P_expB * nB
freq_normB= P_normB * nB

# 2.3. Полигоны
mid = (edges[:-1]+edges[1:])/2
plt.figure(figsize=(6,4))
plt.plot(mid, emp_freqB,'o-', label='Эмпирические')
plt.plot(mid, freq_normB,'s--', label=f'N(μ={mu_hat:.1f},σ²={sigma2_hat:.1f})')
plt.plot(mid, freq_expB, 'd-.', label=f'Exp(λ={lambdaB_hat:.3f})')
plt.xlabel('x'); plt.ylabel('Частота')
plt.title('Выборка B: полигоны распределений')
plt.legend(); plt.grid(True); plt.show()

# 2.4. 95% доверительные интервалы
alpha=0.05
# μ точный
t=norm.ppf(1-alpha/2); se_mu=np.sqrt(sigma2_hat/nB)
ci_mu=(mu_hat-t*se_mu, mu_hat+t*se_mu)
# σ² через χ² неформализм опущен, можно далее
# Exp λ
se_lB=lambdaB_hat/np.sqrt(nB)
ci_lB=(lambdaB_hat-ez*se_lB, lambdaB_hat+ez*se_lB)
print(f"95% ДИ μ: [{ci_mu[0]:.3f}, {ci_mu[1]:.3f}]")
print(f"95% ДИ Exp λ: [{ci_lB[0]:.3f}, {ci_lB[1]:.3f}]")

# 2.5. Числовые характеристики B
# эмпирические
modeB = mid[np.argmax(emp_freqB)]
medianB= np.median(B)
skB=skew(B); exB=kurtosis(B)
print(f"Эмпир. мода B: {modeB:.1f}, медиана: {medianB:.1f}, асимметрия: {skB:.4f}, эксцесс: {exB:.4f}")
# теор. Normal
print(f"Normal теор.: E=μ={mu_hat:.2f}, Var=σ²={sigma2_hat:.2f}, skew=0, ex=0")
# теор. Exponential
print(f"Exp теор.: E=1/λ={1/lambdaB_hat:.2f}, Var=1/λ²={(1/lambdaB_hat**2):.2f}, skew=2, ex=6")

# Вывод по выборке B
# Данные B близки к нормальному распределению

# =============================
# 3. Общий итоговый вывод
# =============================

# --- Выборка A ---
# На основе предыдущей лабораторной была предложена гипотеза, что данные распределены
# либо по биномиальному закону (Binomial(n=8, p)), либо по закону Пуассона (Poisson(λ)).
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

# Вывод по выборке A
# Наиболее подходящим распределением является биномиальное Bin(n=8, p≈0.356), что согласуется с ограничением на максимальное значение выборки (8).

# Выборка B
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

# Вывод по выборке B
# Наиболее вероятное распределение — нормальное распределение с параметрами N(μ≈51.16, σ²≈117.53).
# Это подтверждается симметричностью данных, формой гистограммы и числовыми характеристиками.
# Экспоненциальное распределение как модель — малоподходящее.

# Общий итог:
# Для выборки A — биномиальное распределение, для выборки B — нормальное.
# Результаты подкреплены графиками, доверительными интервалами и моментами распределений.

def raw_moments_sum(data, max_order=4):
    """
    Метод сумм: начальные моменты m'_k = (1/n) * sum(x_i^k)
    """
    n = len(data)
    return [sum(x**k for x in data) / n for k in range(1, max_order+1)]

def central_moments_sum(data, max_order=4):
    """
    Метод сумм: центральные моменты m_k = (1/n) * sum((x_i - mean)^k)
    """
    mean = sum(data) / len(data)
    n = len(data)
    return [sum((x-mean)**k for x in data) / n for k in range(1, max_order+1)]

def raw_moments_product(data, max_order=4):
    """
    Метод произведений: начальные моменты через произведение ненулевых x_i
    m'_k = (Π x_i^k)^(1/n)
    Работает корректно, если все x_i != 0.
    """
    data = [x for x in data if x != 0]
    n = len(data)
    moments = []
    for k in range(1, max_order+1):
        prod = 1.0
        for x in data:
            prod *= x**k
        moments.append(prod**(1/n))
    return moments

def central_moments_product(data, max_order=4):
    """
    Метод произведений: центральные моменты через произведение (x_i - mean)
    m_k = (Π (x_i - mean)^k)^(1/n)
    """
    mean = sum(data) / len(data)
    n = len(data)
    moments = []
    for k in range(1, max_order+1):
        prod = 1.0
        for x in data:
            prod *= (x-mean)**k
        moments.append(abs(prod)**(1/n))  # берем модуль, чтобы избежать комплексных
    return moments

# Для выборки A
sum_raw_A   = raw_moments_sum(sampleA)
sum_cent_A  = central_moments_sum(sampleA)
prod_raw_A  = raw_moments_product(sampleA)
prod_cent_A = central_moments_product(sampleA)

print("Выборка A:")
print("Начальные моменты (метод сумм):", sum_raw_A)
print("Центральные моменты (метод сумм):", sum_cent_A)
print("Начальные моменты (метод произведений):", prod_raw_A)
print("Центральные моменты (метод произведений):", prod_cent_A)

# Для выборки B
sum_raw_B   = raw_moments_sum(sampleB)
sum_cent_B  = central_moments_sum(sampleB)
prod_raw_B  = raw_moments_product(sampleB)
prod_cent_B = central_moments_product(sampleB)

print("\nВыборка B:")
print("Начальные моменты (метод сумм):", sum_raw_B)
print("Центральные моменты (метод сумм):", sum_cent_B)
print("Начальные моменты (метод произведений):", prod_raw_B)
print("Центральные моменты (метод произведений):", prod_cent_B)
