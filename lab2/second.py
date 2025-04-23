"""Second Matstat lab"""
"""Task 1"""
import matplotlib.pyplot as plt
import numpy as np
from collections import *
from scipy.stats import *
import pandas as pd
from scipy.optimize import *
from scipy.special import *
from math import factorial
from math import exp as math_exp

# Данные выборки
data = [
    1, 0, 1, 1, 1, 2, 0, 2, 1, 0, 0, 0, 1, 0, 3, 2, 1, 1, 1, 0,
    0, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 1, 0, 0, 1, 0, 1, 1,
    0, 2, 3, 1, 0, 3, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1, 3, 0, 2, 3,
    2, 1, 1, 0, 4, 2, 2, 1, 1, 2, 0
]

# Начальные моменты до 4-го порядка
mean_val = np.mean(data)
raw_moments = [np.mean(np.power(data, i)) for i in range(1, 5)]

# Центральные моменты до 4-го порядка
central_moments = [np.mean((data - mean_val) ** i) for i in range(1, 5)]

# Вывод результатов
moment_summary = f"""
Начальные моменты:
1-й порядок (среднее): {raw_moments[0]:.3f}
2-й порядок: {raw_moments[1]:.3f}
3-й порядок: {raw_moments[2]:.3f}
4-й порядок: {raw_moments[3]:.3f}

Центральные моменты:
1-й порядок: {central_moments[0]:.3f} (всегда 0)
2-й порядок (дисперсия): {central_moments[1]:.3f}
3-й порядок: {central_moments[2]:.3f}
4-й порядок: {central_moments[3]:.3f}
"""

# print(moment_summary)

# Строим статистический ряд
counter = Counter(data)
sorted_data = sorted(counter.items())

# Разбиваем значения и частоты
values, frequencies = zip(*sorted_data)

# Минимум, максимум и размах
min_val = min(data)
max_val = max(data)
range_val = max_val - min_val
# print(f'Максимум: {max_val}, минимум: {min_val}, размах: {range_val}\n')

# Выводим статистический ряд
# print("Статистический ряд:")
# for value, freq in sorted_data:
#     print(f"{value}: {freq}")

# Рисуем полигон
# plt.figure(figsize=(10, 6))
# plt.plot(values, frequencies, marker='o', linestyle='-', color='b')
# plt.xlabel('Значение выборки')
# plt.ylabel('Частота')
# plt.title('Полигон статистического ряда')
# plt.grid()
# plt.xlim(0, 4)
# plt.ylim(0, 35)
# plt.show()

x = [-1, 0, 1, 2, 3, 4, 5]
y = [0, 0, 21 / 71, 50 / 71, 63 / 71, 70 / 71, 1]

# plt.plot(x, y, drawstyle='steps-pre')
# plt.xlim(-1, 5)
# plt.ylim(0, 1.2)
# plt.grid()
# plt.xlabel("x")
# plt.ylabel("F(x)")
# plt.title("Эмпирическая функция")
# plt.show()

# Кумулята
# cumulative_frequencies = np.cumsum(frequencies)
# plt.figure(figsize=(10, 6))
# plt.plot(values, cumulative_frequencies, marker='o', color='g', linestyle='-')
# plt.xlabel('Значение выборки')
# plt.ylabel('Накопленная частота')
# plt.title('Кумулята')
# plt.grid()
# plt.show()

# Огива
# cumulative_frequencies = np.cumsum(frequencies)
# plt.figure(figsize=(10, 6))
# plt.plot(cumulative_frequencies, values, marker='o', color='g', linestyle='-')
# plt.xlabel('Значение выборки')
# plt.ylabel('Накопленная частота')
# plt.title('Огива')
# plt.grid()
# plt.show()

# Ящик с усами
# plt.figure(figsize=(8, 6))
# plt.boxplot(data, vert=False, patch_artist=True)
# plt.xlabel('Значение выборки')
# plt.title('Ящик с усами')
# plt.grid()
# plt.show()

# Мода (наиболее частое значение)
mode = mode(data).mode

# Медиана
median = np.median(data)

# Коэффициент асимметрии
skewness = skew(data)

# Эксцесс (избыток)
kurtosis = kurtosis(data)

# Выводы
summary = f"""
Мода: {mode}
Медиана: {median}
Коэффициент асимметрии: {skewness:.3f}
Эксцесс: {kurtosis:.3f}

Выводы:
- Эксцесс {kurtosis:.3f} (отрицательный) предполагает, что распределение более плоское, чем нормальное.
- Положительная асимметрия указывает на скошенность вправо: у распределения есть «длинный хвост» в сторону больших значений (2, 3, 4). Это означает, что хотя большинство значений сосредоточено на 0 и 1, иногда могут встречаться и более крупные значения.

Гипотеза:
- По форме напоминает положительно скошенное распределение (например, Пуассоновское или биномиальное).
- Возможная модель: распределение Пуассона, которое часто описывает число событий за фиксированный интервал времени или пространства. Например, количество ошибок или обращений в систему.

Оценка параметров:
- Если предположить распределение Пуассона, то параметр λ можно оценить как среднее выборки: 1.13"""

# print(summary)



print('-' * 50, '\n\n')

n = len(data)
mean_val = np.mean(data)
var_val = np.var(data, ddof=0)

lambda_est = mean_val

print("Оценка параметров для Пуассоновского распределения:")
print(f"Метод моментов/максимального правдоподобия: λ = {lambda_est:.3f}")

# Генерим теоретические вероятности
x_vals = np.arange(0, 6)
poisson_probs = poisson.pmf(x_vals, lambda_est)

# 2. Гипотеза: Биномиальное распределение
# Оценка параметров методом моментов
p_est = 1 - (var_val / mean_val)
n_est = round(mean_val / p_est)

print("Оценка параметров для Биномиального распределения:")
print(f"Метод моментов: n = {n_est}, p = {p_est:.3f}")

# Альтернативная оценка (фиксируем n=max значение в данных)
n_fixed = max(data)
p_fixed = mean_val / n_fixed
# print(f"Альтернативная оценка (n фиксировано): n = {n_fixed}, p = {p_fixed:.3f}")

print('\n\n\n')


counts = Counter(data)
max_val = max(counts.keys())
empirical_freq = [counts.get(i, 0) for i in range(max_val + 1)]

print("+" + "-"*23 + "+")
print("| Значение | Частота |")
print("+" + "-"*23 + "+")

for value, freq in enumerate(empirical_freq):
    print(f"|{value:^9}|{freq:^9}|")

print("+" + "-"*23 + "+")
print('\n')

# 1. Теоретические частоты для Пуассона (формула)
def poisson_prob(k, lambd):
    """Формула Пуассона P(X=k) = (λ^k * e^-λ) / k!"""
    return (lambd**k * math_exp(-lambd)) / factorial(k)

lambda_est = mean_val
poisson_probs = [poisson_prob(k, lambda_est) for k in range(max_val + 1)]
poisson_freq = [round(p * n) for p in poisson_probs]

print("Пуассоновское распределение:")
print(f"λ = {lambda_est:.3f}")
print("k\tТеор. вероятность\tТеор. частота")
for k in range(max_val + 1):
    print(f"{k}\t{poisson_probs[k]:.4f}\t\t\t{poisson_freq[k]}")




# 2. Теоретические частоты для Бинома (формула)
def binomial_prob(k, n, p):
    """Формула Бинома P(X=k) = C(n,k) * p^k * (1-p)^(n-k)"""
    n = int(round(n))
    comb = factorial(n) // (factorial(k) * factorial(n - k))
    return comb * (p**k) * ((1 - p)**(n - k))


binom_probs = [binomial_prob(k, n_est, p_est) for k in range(max_val + 1)]
binom_freq = [round(p * n) for p in binom_probs]

print("\nБиномиальное распределение:")
print(f"n = {n_est}, p = {p_est:.3f}")
print("k\tТеор. вероятность\tТеор. частота")
for k in range(max_val + 1):
    print(f"{k}\t{binom_probs[k]:.4f}\t\t\t{binom_freq[k]}")



x_values = np.arange(max_val + 1)

plt.figure(figsize=(12, 6))

# Полигон эмпирических частот
plt.plot(x_values, empirical_freq, 'b-o',
         linewidth=2, markersize=8, label='Эмпирическое распределение')

# Полигон Пуассона
plt.plot(x_values, poisson_freq, 'r--s',
         linewidth=2, markersize=6, label=f'Пуассон (λ={lambda_est:.2f})')

# Полигон Бинома
plt.plot(x_values, binom_freq, 'g-.D',
         linewidth=2, markersize=6, label=f'Бином (n={n_est}, p={p_est:.2f})')

# Настройки графика
plt.title('Сравнение эмпирического и теоретических распределений', fontsize=14)
plt.xlabel('Значение', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.xticks(x_values)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Добавление значений точек
for x, y in zip(x_values, empirical_freq):
    plt.text(x, y+0.5, f'{y}', ha='center', va='bottom', color='blue')

for x, y in zip(x_values, poisson_freq):
    plt.text(x, y+0.5, f'{y:.1f}', ha='center', va='bottom', color='red')

for x, y in zip(x_values, binom_freq):
    plt.text(x, y+0.5, f'{y:.1f}', ha='center', va='bottom', color='green')

plt.show()



# Вычисление начальных моментов методом сумм
def raw_moments_sum(data, max_order):
    n = len(data)
    return [sum(x**k for x in data)/n for k in range(1, max_order+1)]

# Вычисление центральных моментов методом сумм
def central_moments_sum(data, max_order):
    mean = np.mean(data)
    n = len(data)
    return [sum((x-mean)**k for x in data)/n for k in range(1, max_order+1)]


# Вычисление начальных моментов методом произведений
def raw_moments_product(data, max_order=4):
    data = np.array(data, dtype=np.float64)
    non_zero = data[data != 0]
    n_total = len(data)

    if len(non_zero) == 0:
        return [0.0] * max_order

    moments = []
    for k in range(1, max_order + 1):
        prod = np.prod(non_zero ** k, dtype=np.float64)
        moment = prod ** (1 / n_total)
        moments.append(moment)

    return moments

# Вычисление центральных моментов методом произведений
def central_moments_product(data, max_order):
    mean = np.mean(data)
    n = len(data)
    moments = []
    for k in range(1, max_order+1):
        prod = 1
        for x in data:
            prod *= (x-mean)**k
        moments.append(prod**(1/n))
    return moments


# Вычисляем моменты
max_order = 4
sum_raw = raw_moments_sum(data, max_order)
prod_raw = raw_moments_product(data, max_order)
sum_cent = central_moments_sum(data, max_order)
prod_cent = central_moments_product(data, max_order)

# Сравниваем результаты
print("\n\n\nСравнение методов вычисления моментов:")
print("\nНачальные моменты:")
print(f"{'Порядок':<8} {'Метод сумм':<15} {'Метод произвед.':<15} {'Разница':<10}")
for k in range(max_order):
    diff = np.abs(sum_raw[k] - prod_raw[k])
    print(f"{k+1:<8} {sum_raw[k]:<15.6f} {prod_raw[k]:<15.6f} {diff:.6f}")

print("\nЦентральные моменты:")
print(f"{'Порядок':<8} {'Метод сумм':<15} {'Метод произвед.':<15} {'Разница':<10}")
for k in range(max_order):
    diff = np.abs(sum_cent[k] - prod_cent[k])
    print(f"{k+1:<8} {sum_cent[k]:<15.6f} {prod_cent[k]:<15.6f} {diff:.6f}")



print('\n\nДоверительные интервалы:')
lambda_hat = np.mean(data)

se_lambda = np.sqrt(lambda_hat / n)  # Standard error (корень из lambda\n)
z = norm.ppf(0.975)  # Квантиль СНО, равен 1.96 для 95% ДИ

ci_lower = lambda_hat - z * se_lambda
ci_upper = lambda_hat + z * se_lambda

print(f"95% ДИ для λ: [{ci_lower:.4f}, {ci_upper:.4f}]")




n_estimated = max(data) if max(data) != 0 else 1  # Защита от деления на 0
p_hat = np.mean(data) / n_estimated  # Оценка параметра p

# 2. Arcsin-преобразование, для стабилизации дисперсии
theta_hat = np.arcsin(np.sqrt(p_hat))  # Арксинус из корня
se_theta = 1 / (2 * np.sqrt(len(data)))

# 3. Квантиль нормального распределения
z = norm.ppf(0.975)  # z ~ 1.96

# 4. Доверительный интервал для theta
ci_theta_lower = theta_hat - z * se_theta
ci_theta_upper = theta_hat + z * se_theta

# 5. Обратное преобразование в p
ci_p_lower = np.sin(ci_theta_lower)**2
ci_p_upper = np.sin(ci_theta_upper)**2

# Вывод
print(f"95% ДИ для p: [{ci_p_lower:.4f}, {ci_p_upper:.4f}]")





# Эмпирические характеристики
empirical_mean = mean_val
empirical_var = var_val
empirical_mode = Counter(data).most_common(1)[0][0]
empirical_median = np.median(data)
empirical_skew = skew(data)
empirical_kurtosis = kurtosis

# Теоретические характеристики для Пуассона
poisson_mean = lambda_est
poisson_var = lambda_est
poisson_mode = np.floor(lambda_est)
poisson_median = lambda_est - 1/3
poisson_skew = 1/np.sqrt(lambda_est)
poisson_kurtosis = 1/lambda_est

# Теоретические характеристики для Бинома
binom_mean = n_est * p_est
binom_var = n_est * p_est * (1 - p_est)
binom_mode = np.floor((n_est + 1) * p_est)
binom_median = np.floor(n_est * p_est)
binom_skew = (1 - 2*p_est)/np.sqrt(n_est*p_est*(1 - p_est))
binom_kurtosis = (1 - 6*p_est*(1 - p_est))/(n_est*p_est*(1 - p_est))

# Вывод результатов
print("СРАВНЕНИЕ ХАРАКТЕРИСТИК РАСПРЕДЕЛЕНИЙ")
print("\n1. Пуассоновское распределение (Poisson(λ))")
print(f"Теоретические характеристики:")
print(f"λ = {lambda_est:.3f}")
print(f"Математическое ожидание: {poisson_mean:.3f}")
print(f"Дисперсия: {np.var(data):.3f}")
print(f"Мода: {poisson_mode}")
print(f"Медиана: ≈ {poisson_median:.3f}")
print(f"Коэффициент асимметрии: {poisson_skew:.3f}")
print(f"Коэффициент эксцесса: {poisson_kurtosis:.3f}")

print("\n2. Биномиальное распределение (Bin(n,p))")
print(f"Теоретические характеристики:")
print(f"n = {n_est}, p = {p_est:.3f}")
print(f"Математическое ожидание: {binom_mean:.3f}")
print(f"Дисперсия: {binom_var:.3f}")
print(f"Мода: {binom_mode}")
print(f"Медиана: ≈ {binom_median}")
print(f"Коэффициент асимметрии: {binom_skew:.3f}")
print(f"Коэффициент эксцесса: {binom_kurtosis:.3f}")

print("\n3. Эмпирические характеристики:")
print(f"Выборочное среднее: {empirical_mean:.3f}")
print(f"Выборочная дисперсия: {empirical_var:.3f}")
print(f"Мода: {empirical_mode}")
print(f"Медиана: {empirical_median}")
print(f"Асимметрия: {empirical_skew:.3f}")
print(f"Эксцесс: {empirical_kurtosis:.3f}")




# метод сумм и метод произведений сравнить, найти моменты
# доверительные интервалы


"""Task 2"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

# Данные
sample = np.array([
    -71, -74, -51, -87, -37, -72, -64, -77, -63, -58, -50, -39, -71, -48, -39,
    -44, -62, -59, -50, -91, -51, -65, -54, -70, -50, -46, -66, -44, -71, -72,
    -58, -52, -58, -72, -62, -47, -58, -69, -58, -72, -42, -70, -84, -65, -65,
    -49, -45, -53, -50, -73, -47, -21, -40, -74, -35, -49, -58, -44, -50, -74,
    -53, -66, -67, -64, -62, -49, -67, -41, -67, -57, -56, -60, -71, -46, -79,
    -72, -48, -38, -51, -37, -60, -58, -52, -58, -55, -50, -64, -68, -53, -70,
    -59, -59, -90, -72, -61, -74, -36, -79, -65, -68, -39, -54, -86, -49, -48,
    -59, -44, -68, -52, -60, -42, -78, -29, -68, -79, -78, -68, -70, -66, -45,
    -62, -56, -36, -67, -64, -45, -63, -59, -59, -58, -67, -69, -44, -69, -69,
    -78, -58, -53, -58, -71, -52, -59, -27, -46, -56, -72, -50, -55, -30, -56,
    -62, -34, -35, -38, -64, -67, -59, -59, -58, -62, -88, -30, -41, -59, -54,
    -47, -55, -50, -79, -63, -50, -77, -23, -73, -63, -63, -67, -74, -59, -60,
    -46, -57, -74, -54
])

# # Вычисление начальных и центральных моментов до 4-го порядка
# mean = np.mean(sample)
# n = len(sample)
#
# initial_moments = [np.mean(sample ** k) for k in range(1, 5)]
# central_moments = [np.mean((sample - mean) ** k) for k in range(1, 5)]
#
# # Вывод моментов
# for i in range(4):
#     print(f"Начальный момент {i+1}-го порядка: {initial_moments[i]:.2f}")
#     print(f"Центральный момент {i+1}-го порядка: {central_moments[i]:.2f}\n")


min_val = min(sample)
max_val = max(sample)
range_val = max_val - min_val
# print(f'Максимум: {max_val}, минимум: {min_val}, размах: {range_val}\n')


# Определение количества интервалов и длины интервала
n = len(sample)
k = int(np.sqrt(n))  # Правило Стерджеса

interval_length = (max_val - min_val) / k

# print(f"Количество интервалов: {k}")
# print(f"Длина интервала: {interval_length:.2f}")


# Построение интервального ряда
intervals = pd.cut(sample, bins=k)
frequency_table = pd.value_counts(intervals, sort=False)
# frequency_table = pd.Series().value_counts(intervals, sort=False)

# print("\nИнтервальный ряд и частоты:")
# print(frequency_table)

lx = [-500]
ly = [0, 0]
s = ly[0]
for key, value in frequency_table.items():
    lx.append(key.left)
    s += value
    ly.append(s / 184)
lx.append(-21.0)
lx.append(-21)
ly.append(1)
lx.append(0)
ly.append(1)

# 3. Построение гистограммы и полигона
# plt.figure(figsize=(12, 6))
#
# # Гистограмма
# plt.hist(sample, bins=k, edgecolor='black', alpha=0.7, label='Гистограмма')
#
# # Полигон
# midpoints = [(interval.mid) for interval in frequency_table.index]
# frequencies = frequency_table.values
# plt.plot(midpoints, frequencies, marker='o', color='red', label='Полигон')
#
# plt.xlabel('Значение')
# plt.ylabel('Частота')
# plt.title('Гистограмма и полигон распределения')
# plt.axvline(-58, linestyle='-.', color='black', label=f'Мода: -58')
# plt.legend()
# plt.grid()
# plt.show()


# Определение интервалов
n_intervals = int(np.sqrt(len(sample)))

# Построение гистограммы и полигона
hist, bins = np.histogram(sample, bins=n_intervals)

# Эмпирическая функция распределения
sorted_sample = np.sort(sample)

# plt.figure(figsize=(10, 6))
# plt.plot(lx, ly, marker='o', linestyle='-', color='b')
# plt.xlabel('x')
# plt.ylabel('F(x)')
# plt.title('Эмпирическая функция распределения')
# plt.xlim(-95, -15)
# plt.xticks(np.arange(-95, -15, 10))
# plt.ylim(0, 1.2)
# plt.axvline(-59, label='Медиана: -59.0', linestyle='-.')
# plt.legend()
# plt.grid()
# plt.show()

# Кумулята
# plt.figure(figsize=(10, 6))
# plt.plot(lx, ly, marker='*', linestyle='-', color='green')
# plt.xlabel('x')
# plt.ylabel('F(x)')
# plt.title('Кумулята')
# plt.xlim(-100, 0)
# plt.xticks(np.arange(-100, 0, 5))
# plt.ylim(0, 1.05)
# plt.grid()
# plt.show()

# Огива
# plt.figure(figsize=(10, 6))
# plt.plot(ly, lx, marker='*', linestyle='-', color='green')
# plt.xlabel('x')
# plt.ylabel('F(x)')
# plt.title('Огива')
# plt.grid()
# plt.yticks(np.arange(-100, -10, 5))
# plt.xlim(0, 1.05)
# plt.ylim(-100, -15)
# plt.show()

# Ящик с усами
# plt.figure(figsize=(8, 6))
# plt.boxplot(sample, vert=False, patch_artist=True)
# plt.title('Ящик с усами')
# plt.xlabel('Значение')
# plt.grid()
# plt.show()

# Вычисление моды и медианы
mode = -58
median = np.median(sample)

# Коэффициенты асимметрии и эксцесса
# asymmetry = skew(sample)
# kurt = kurtosis(sample)

# Вывод результатов
# print(f'Мода: {mode}')
# print(f'Медиана: {median}')
# print(f'Коэффициент асимметрии: {asymmetry:.2f}')
# print(f'Коэффициент эксцесса: {kurt:.2f}')

# На графике гистограммы отметим моду и медиану
# plt.figure(figsize=(12, 6))
# plt.bar(bins[:-1], hist, width=interval_length, alpha=0.7, label='Гистограмма')
# plt.axvline(mode, color='red', linestyle='--', label=f'Мода: {mode}')
# plt.axvline(median, color='purple', linestyle='-.', label=f'Медиана: {median}')
# plt.xlabel('Интервалы')
# plt.ylabel('Частота')
# plt.legend()
# plt.grid()
# plt.show()

# Выводы
# print(f"""
# - Выводы:
#
# Форма распределения: асимметричная
# Наибольшая частота: видно несколько пиков в районе -58, -50, -59, -72
# Эмпирическая функция: достаточно плавная, но есть резкие скачки, что отражает наличие повторяющихся значений
#
# - Гипотезы о генеральной совокупности:
#
# Тип распределения: По форме графиков и коэффициентам — распределение похоже на нормальное.
# Приблизительные значения: mu (среднее)  = -59, sigma (СКО) = 13.2.
# Такой профиль характерен для процессов с множеством факторов влияния, где значения группируются вокруг среднего, но с выбросами""")




print('-' * 50)



# Гипотеза 1: Нормальное распределение N(μ, σ²)
print("Гипотеза 1: Нормальное распределение N(μ, σ²)")

# Метод моментов:
mu_mm = np.mean(sample)
sigma2_mm = np.var(sample, ddof=0)
print(f"Метод моментов: μ = {mu_mm:.2f}, σ² = {sigma2_mm:.2f}")

# Метод максимального правдоподобия:
mu_mle, sigma_mle = norm.fit(sample)
print(f"Метод максимального правдоподобия: μ = {mu_mle:.2f}, σ = {sigma_mle:.2f}")

# Гипотеза 2: Распределение Лапласа Laplace(μ, b)
print("\nГипотеза 2: Распределение Лапласа Laplace(μ, b)")

# Метод моментов:
mu_laplace = np.median(sample)  # Для Лапласа медиана = μ
b_laplace = np.mean(np.abs(sample - mu_laplace))  # Масштабный параметр
print(f"Метод моментов: μ = {mu_laplace:.2f}, b = {b_laplace:.2f}")

# Метод максимального правдоподобия:
# Для Лапласа оценки ММП совпадают с методом моментов
print(f"Метод максимального правдоподобия: μ = {mu_laplace:.2f}, b = {b_laplace:.2f}")

# Сравнение теоретических частот
hist, bin_edges = np.histogram(sample, bins=int(np.sqrt(len(sample))))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Нормальное распределение
norm_probs = norm.cdf(bin_edges[1:], mu_mle, sigma_mle) - norm.cdf(bin_edges[:-1], mu_mle, sigma_mle)
norm_freq = norm_probs * len(sample)

# Распределение Лапласа
laplace_probs = laplace.cdf(bin_edges[1:], loc=mu_laplace, scale=b_laplace) - \
                laplace.cdf(bin_edges[:-1], loc=mu_laplace, scale=b_laplace)
laplace_freq = laplace_probs * len(sample)

# Таблица сравнения
comparison_table = pd.DataFrame({
    'Интервал': [f"({bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]" for i in range(len(bin_edges)-1)],
    'Эмпирич. частота': hist,
    'Нормальное': norm_freq.round(1),
    'Лапласа': laplace_freq.round(1)
})

print("\nСравнение теоретических частот:")
print(comparison_table.to_string(index=False))

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(bin_centers, hist, 'bo-', label='Эмпирическое', linewidth=2)
plt.plot(bin_centers, norm_freq, 'g^-', label=f'Нормальное (μ={mu_mle:.1f}, σ={sigma_mle:.1f})', linewidth=2)
plt.plot(bin_centers, laplace_freq, 'mD--', label=f'Лапласа (μ={mu_laplace:.1f}, b={b_laplace:.1f})', linewidth=2)

plt.title('Сравнение эмпирического распределения с теоретическими', fontsize=14)
plt.xlabel('Значение', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()



# Метод сумм (корректный для любых данных)
def method_sums(data):
    m1 = np.mean(data)
    m2 = np.mean(data ** 2)

    return {'E[X]': m1, 'E[X²]': m2}


# Метод произведений (положительные данные)
def method_products(data):
    if np.any(data <= 0):
        return None

    log_data = np.log(data)  # Логарифм
    lm1 = np.mean(log_data)
    lm2 = np.var(log_data)

    m1 = np.exp(lm1 + lm2 / 2)  # Экспонента
    m2 = np.exp(2 * lm1 + 2 * lm2)

    return {'E[X]': m1, 'E[X²]': m2}


# Сравнение
results_sum = method_sums(sample)

print("\n\nМетод сумм (рекомендуется):")
for k, v in results_sum.items():
    print(f"{k}: {v:.4f}")

print("\nМетод произведений:")
prod_results = method_products(sample + 92)
if prod_results:
    for k, v in prod_results.items():
        print(f"{k}: {v:.4f}")



# Точечные оценки
mu_hat = np.mean(sample)
sigma_hat = np.std(sample, ddof=1)
n = len(sample)

# 95% ДИ для μ (точный)
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df=n-1)
ci_mu = (mu_hat - t_crit*sigma_hat/np.sqrt(n),
         mu_hat + t_crit*sigma_hat/np.sqrt(n))

# 95% ДИ для σ² (точный)
chi2_low = chi2.ppf(1 - alpha/2, df=n-1)
chi2_high = chi2.ppf(alpha/2, df=n-1)
ci_sigma2 = ((n-1)*sigma_hat**2/chi2_low,
             (n-1)*sigma_hat**2/chi2_high)

print("\n\n\nНормальное распределение:")
print(f"μ ∈ ({ci_mu[0]:.2f}, {ci_mu[1]:.2f})")
print(f"σ² ∈ ({ci_sigma2[0]:.2f}, {ci_sigma2[1]:.2f})")




# Точечные оценки
mu_hat_laplace = np.median(sample)
b_hat = np.mean(np.abs(sample - mu_hat_laplace))

# Асимптотические 95% ДИ (используем ЗБЧ)
se_mu = b_hat/np.sqrt(n)
se_b = b_hat*np.sqrt(2/n)

ci_mu_laplace = (mu_hat_laplace - norm.ppf(0.975)*se_mu,
                 mu_hat_laplace + norm.ppf(0.975)*se_mu)

ci_b = (b_hat - norm.ppf(0.975)*se_b,
        b_hat + norm.ppf(0.975)*se_b)

print("\nРаспределение Лапласа:")
print(f"μ ∈ ({ci_mu_laplace[0]:.2f}, {ci_mu_laplace[1]:.2f})")
print(f"b ∈ ({ci_b[0]:.2f}, {ci_b[1]:.2f})")




# Эмпирические характеристики
empirical_mean = np.mean(sample)
empirical_median = np.median(sample)
empirical_mode = float(max(set(sample.tolist()), key=sample.tolist().count))
empirical_var = np.var(sample, ddof=0)
empirical_skew = skew(sample)
empirical_kurtosis = kurtosis(sample, fisher=False)

# Параметры распределений
mu_norm, sigma_norm = norm.fit(sample)
mu_laplace = np.median(sample)
b_laplace = np.mean(np.abs(sample - mu_laplace))
shift = -min(sample) + 1e-6
lambda_exp = 1/np.mean(sample + shift)

# Создаем таблицу
from tabulate import tabulate

table = [
    ["Характеристика", "Нормальное (теор.)", "Лапласа (теор.)", "Эмпирич."],
    ["Мода", f"μ = {mu_norm:.2f}", f"μ = {mu_laplace:.1f}", f"{empirical_mode:.1f}"],
    ["Медиана", f"{mu_norm:.2f}", f"{mu_laplace:.1f}", f"{empirical_median:.1f}"],
    ["Мат. ожидание", f"{mu_norm:.2f}", f"{mu_laplace:.1f}", f"{empirical_mean:.2f}"],
    ["Дисперсия", f"{sigma_norm**2:.2f}", f"{2*b_laplace**2:.2f}", f"{empirical_var:.2f}"],
    ["Асимметрия", "0", "0", f"{empirical_skew:.2f}"],
    ["Эксцесс", "0", "3", f"{empirical_kurtosis:.2f}"]
]

print(tabulate(table, headers="firstrow", tablefmt="grid", stralign="center"))
