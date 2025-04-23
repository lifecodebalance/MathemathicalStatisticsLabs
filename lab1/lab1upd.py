import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

# ============================
# Работа с выборкой A
# ============================

# Данные выборки A
data_A = np.array([
    5, 3, 3, 3, 5, 4, 5, 3, 3, 4, 2, 1, 5,
    2, 4, 0, 2, 2, 3, 2, 1, 3, 3, 1, 2, 4,
    6, 6, 4, 1, 2, 4, 3, 1, 5, 2, 4, 6, 3,
    8, 4, 5, 1, 1, 2, 0, 2, 3, 3, 2, 4, 2,
    1, 2, 3, 1, 2, 4, 3, 0, 6, 3, 1, 4, 3, 7,
    1, 1, 0, 2, 3, 1, 1
])

# 1. Вычисление минимального и максимального значения, а также размаха выборки A
min_A = np.min(data_A)
max_A = np.max(data_A)
range_A = max_A - min_A

print("Выборка A:")
print("Минимальное значение:", min_A)
print("Максимальное значение:", max_A)
print("Размах выборки:", range_A)

# 2. Построение гистограммы для выборки A
plt.figure(figsize=(8,5))
# Определяем интервалы так, чтобы каждый отдельный дискретный элемент оказался в отдельном интервале
bins_A = np.arange(min_A - 0.5, max_A + 1.5, 1)
plt.hist(data_A, bins=bins_A, edgecolor='black', alpha=0.7)
plt.title("Гистограмма для выборки A")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

# 3. Статистический ряд: подсчёт частот для каждого уникального значения
freq_A = pd.Series(data_A).value_counts().sort_index()
print("\nСтатистический ряд (значение: частота):")
print(freq_A)

# 4. Построение полигона частот (график, где точки – это частоты, соединённые линиями)
plt.figure(figsize=(8,5))
plt.plot(freq_A.index, freq_A.values, marker='o', linestyle='-')
plt.title("Полигон частот для выборки A")
plt.xlabel("Значения")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

# 5. Построение эмпирической функции распределения (ЭФР)
n_A = len(data_A)
sorted_A = np.sort(data_A)           # сортировка
edf_y = np.arange(1, n_A+1) / n_A      # вычисляем накопленные доли по формуле

plt.figure(figsize=(8,5))
plt.step(sorted_A, edf_y, where="post")
plt.title("Эмпирическая функция распределения для выборки A")
plt.xlabel("Значения")
plt.ylabel("F(x)")
plt.grid(True)
plt.show()

# 6. Вычисление начальных и центральных эмпирических моментов до 4-го порядка
mean_A = np.mean(data_A)  # первый момент (среднее)
m1 = mean_A
m2 = np.mean(data_A**2)
m3 = np.mean(data_A**3)
m4 = np.mean(data_A**4)

# Центральные моменты (отклонения от среднего)
mu2 = np.mean((data_A - mean_A)**2)  # дисперсия
mu3 = np.mean((data_A - mean_A)**3)
mu4 = np.mean((data_A - mean_A)**4)

print("\nЭмпирические начальные моменты для выборки A:")
print("Первый момент (среднее):", m1)
print("Второй момент:", m2)
print("Третий момент:", m3)
print("Четвертый момент:", m4)

print("\nЭмпирические центральные моменты для выборки A:")
print("Второй центральный момент (дисперсия):", mu2)
print("Третий центральный момент:", mu3)
print("Четвертый центральный момент:", mu4)

# 7. Нахождение моды и медианы, а также коэффициентов асимметрии и эксцесса
mode_A_result = stats.mode(data_A)
# Гарантируем, что mode_A_result.mode и .count - массивы, и берем первый элемент
mode_A_value = np.atleast_1d(mode_A_result.mode)[0]
mode_A_count = np.atleast_1d(mode_A_result.count)[0]
median_A = np.median(data_A)
skewness_A = stats.skew(data_A, bias=False)
kurtosis_A = stats.kurtosis(data_A, bias=False)  # эксцесс (избыточный куртозис)

print("\nМода для выборки A:", mode_A_value, "с частотой", mode_A_count)
print("Медиана для выборки A:", median_A)
print("Коэффициент асимметрии (skewness):", skewness_A)
print("Эксцесс (excess kurtosis):", kurtosis_A)

# 8. Построение ящика с усами (boxplot)
plt.figure(figsize=(4,6))
plt.boxplot(data_A, vert=True)
plt.title("Ящик с усами для выборки A")
plt.ylabel("Значения")
plt.grid(True)
plt.show()

# 9. Построение кумуляты (накопленной частоты)
cumulative_counts_A = np.cumsum(freq_A.values)
plt.figure(figsize=(8,5))
plt.step(freq_A.index, cumulative_counts_A, where="post")
plt.title("Кумулята для выборки A")
plt.xlabel("Значения")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.show()

# 10. Огива для выборки A
cumulative_counts_A = np.cumsum(freq_A.values)
plt.figure(figsize=(8,5))
plt.step(freq_A.index, cumulative_counts_A, where="post")
plt.title("Кумулята (огива) для выборки A")
plt.xlabel("Значения")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.show()

# 11. Выводы и гипотезы для выборки A
print("\nВыводы для выборки A:")
print("Выборка A представляет дискретное распределение.")
print(f"Мода равна {mode_A_value} (наиболее часто встречающееся значение), медиана = {median_A}.")
print(f"Среднее значение равно {mean_A:.2f} и дисперсия {mu2:.2f}.")
if skewness_A > 0:
    asymmetry_text = "правосторонней"
elif skewness_A < 0:
    asymmetry_text = "левосторонней"
else:
    asymmetry_text = "симметричной"
print(f"Коэффициент асимметрии равен {skewness_A:.2f} ({asymmetry_text} асимметрия), эксцесс = {kurtosis_A:.2f}.")
print("Предположительно, генеральная совокупность может быть описана дискретным распределением (например, распределением Пуассона или биномиальным).")

# ============================
# Работа с выборкой B
# ============================

# Данные выборки B
data_B = np.array([
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
])

# 1. Вычисление минимального и максимального значения, а также размаха выборки B
min_B = np.min(data_B)
max_B = np.max(data_B)
range_B = max_B - min_B

print("\nВыборка B:")
print("Минимальное значение:", min_B)
print("Максимальное значение:", max_B)
print("Размах выборки:", range_B)

# 2. Определение оптимального количества интервалов группировки и длины интервала
# Используем правило Стерджеса: k = 1 + log2(n)
n_B = len(data_B)
k = int(np.ceil(1 + np.log2(n_B)))
interval_length = range_B / k

print("Оптимальное количество интервалов (по правилу Стерджеса):", k)
print("Длина интервала группировки:", np.round(interval_length, 2))

# 3. Построение интервального ряда: подсчет частот по интервалам
bins = np.linspace(min_B, max_B, k+1)  # границы интервалов
hist, bin_edges = np.histogram(data_B, bins=bins)
print("\nИнтервальный ряд (частоты по интервалам):")
for i in range(len(hist)):
    print(f"Интервал [{bin_edges[i]}, {bin_edges[i+1]}): {hist[i]}")

# 4. Построение гистограммы и полигона частот для выборки B
plt.figure(figsize=(10,5))
plt.hist(data_B, bins=bins, edgecolor='black', alpha=0.7)
plt.title("Гистограмма для выборки B")
plt.xlabel("Интервалы")
plt.ylabel("Частота")
# Отмечаем моду: выбираем интервал с максимальной частотой
mode_bin_index = np.argmax(hist)
mode_bin = (bin_edges[mode_bin_index], bin_edges[mode_bin_index+1])
plt.axvline(x=(mode_bin[0]+mode_bin[1])/2, color='r', linestyle='--',
            label=f'Мода: [{mode_bin[0]:.1f}, {mode_bin[1]:.1f})')
plt.legend()
plt.show()

# Строим полигон частот: используем середины интервалов (midpoints)
midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.figure(figsize=(10,5))
plt.plot(midpoints, hist, marker='o', linestyle='-')
plt.title("Полигон частот для выборки B")
plt.xlabel("Среднее значение интервала")
plt.ylabel("Частота")
plt.grid(True)
plt.show()

# 5. Построение эмпирической функции распределения для выборки B
sorted_B = np.sort(data_B)
edf_y_B = np.arange(1, n_B+1) / n_B

plt.figure(figsize=(8,5))
plt.step(sorted_B, edf_y_B, where="post")
plt.title("Эмпирическая функция распределения для выборки B")
plt.xlabel("Значения")
plt.ylabel("F(x)")
plt.grid(True)
plt.show()

# 6. Построение кумуляты для выборки B (накопленная частота)
freq_B, edges_B = np.histogram(data_B, bins=bins)
cumulative_counts_B = np.cumsum(freq_B)
plt.figure(figsize=(8,5))
plt.step(bin_edges[1:], cumulative_counts_B, where="post")
plt.title("Кумулята для выборки B")
plt.xlabel("Значения")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.show()

# 7. Вычисление эмпирических моментов до 4-го порядка для выборки B
mean_B = np.mean(data_B)
m1_B = mean_B
m2_B = np.mean(data_B**2)
m3_B = np.mean(data_B**3)
m4_B = np.mean(data_B**4)

mu2_B = np.mean((data_B - mean_B)**2)
mu3_B = np.mean((data_B - mean_B)**3)
mu4_B = np.mean((data_B - mean_B)**4)

print("\nЭмпирические начальные моменты для выборки B:")
print("Первый момент (среднее):", m1_B)
print("Второй момент:", m2_B)
print("Третий момент:", m3_B)
print("Четвертый момент:", m4_B)

print("\nЭмпирические центральные моменты для выборки B:")
print("Второй центральный момент (дисперсия):", mu2_B)
print("Третий центральный момент:", mu3_B)
print("Четвертый центральный момент:", mu4_B)

# 8. Нахождение моды и медианы для выборки B, а также коэффициентов асимметрии и эксцесса
mode_B_result = stats.mode(data_B)
mode_B_value = np.atleast_1d(mode_B_result.mode)[0]
mode_B_count = np.atleast_1d(mode_B_result.count)[0]
median_B = np.median(data_B)
skewness_B = stats.skew(data_B, bias=False)
kurtosis_B = stats.kurtosis(data_B, bias=False)

print("\nМода для выборки B:", mode_B_value, "с частотой", mode_B_count)
print("Медиана для выборки B:", median_B)
print("Коэффициент асимметрии для выборки B:", skewness_B)
print("Эксцесс для выборки B:", kurtosis_B)

# 9. Построение ящика с усами для выборки B
plt.figure(figsize=(4,6))
plt.boxplot(data_B, vert=True)
plt.title("Ящик с усами для выборки B")
plt.ylabel("Значения")
plt.grid(True)
plt.show()

# 10. Отметка медианы на кумуляте для выборки B
plt.figure(figsize=(8,5))
plt.step(bin_edges[1:], cumulative_counts_B, where="post", label="Кумулята")
plt.axvline(x=median_B, color='g', linestyle='--', label=f'Медиана: {median_B}')
plt.title("Кумулята для выборки B с отмеченной медианой")
plt.xlabel("Значения")
plt.ylabel("Накопленная частота")
plt.legend()
plt.grid(True)
plt.show()

# 11. Огива Б
cumulative_counts_B = np.cumsum(hist)
plt.figure(figsize=(8,5))
plt.step(bin_edges[1:], cumulative_counts_B, where="post")
plt.title("Кумулята (огива) для выборки B")
plt.xlabel("Значения")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.show()

# 12. Выводы и гипотезы для выборки B
print("\nВыводы для выборки B:")
print(f"Выборка B имеет диапазон от {min_B} до {max_B} с оптимальным количеством интервалов = {k}.")
print(f"Среднее значение равно {mean_B:.2f}, а медиана = {median_B}.")
if abs(skewness_B) < 0.5:
    asymmetry_text_B = "распределение можно считать приблизительно симметричным"
elif skewness_B > 0:
    asymmetry_text_B = "наблюдается правосторонняя асимметрия"
else:
    asymmetry_text_B = "наблюдается левосторонняя асимметрия"
print(f"Коэффициент асимметрии = {skewness_B:.2f} ({asymmetry_text_B}), а дисперсия = {mu2_B:.2f}.")
if kurtosis_B > 0:
    kurtosis_text_B = "более острые пики"
elif kurtosis_B < 0:
    kurtosis_text_B = "более плоская вершина"
else:
    kurtosis_text_B = "нормальная форма"
print(f"Эксцесс = {kurtosis_B:.2f}, что указывает на {kurtosis_text_B} распределение.")
print("На основании анализа можно предположить, что генеральная совокупность может быть описана нормальным распределением.")

print("\n==================== ЧТО Я СДЕЛАЛ? ====================")
print("Для выборки A:")
print("1. Нашёл максимальный и минимальный элементы, а также размах выборки.")
print("2. Построил статистический ряд и полигон частот.")
print("3. Построил эмпирическую функцию распределения.")
print("4. Вычислил начальные и центральные эмпирические моменты до 4-го порядка.")
print("5. Нашёл мода, медиана, коэффициенты асимметрии и эксцесса.")
print("6. Построил кумулята и ящик с усами (boxplot).")
print("7. Сформулировал выводы и гипотезы о распределении генеральной совокупности.\n")

print("Для выборки B:")
print("1. Нашёл максимальный и минимальный элементы, а также размах выборки.")
print("2. Определил оптимальное количество интервалов группировки и вычислил длина интервала.")
print("3. Построил интервальный ряд, гистограмма и полигон частот.")
print("4. Построил эмпирическую функцию распределения и кумуляту.")
print("5. Вычислил начальные и центральные эмпирические моменты до 4-го порядка.")
print("6. Нашёл моду (отметил на гистограмме) и медиану (отметил на кумуляте), а также коэффициенты асимметрии и эксцесса.")
print("7. Построил ящик с усами (boxplot).")
print("8. Продумал выводы и гипотезы о распределении генеральной совокупности.")
print("====================================================================")