import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Допустим, у вас есть дискретная выборка A (может быть и непрерывная, неважно):
sampleA = [
    5, 3, 3, 3, 5, 4, 5, 3, 3, 4, 2, 1, 5,
    2, 4, 0, 2, 2, 3, 2, 1, 3, 3, 1, 2, 4,
    6, 6, 4, 1, 2, 4, 3, 1, 5, 2, 4, 6, 3,
    8, 4, 5, 1, 1, 2, 0, 2, 3, 3, 2, 4, 2,
    1, 2, 3, 1, 2, 4, 3, 0, 6, 3, 1, 4, 3, 7,
    1, 1, 0, 2, 3, 1, 1
]

dataA = np.array(sampleA)
nA = len(dataA)

# 1) Сортируем выборку
sortedA = np.sort(dataA)

# 2) Оцениваем плотность (KDE)
kdeA = gaussian_kde(sortedA)
# (При желании можно настроить ширину ядра: kdeA = gaussian_kde(sortedA, bw_method=...)

# 3) Строим сетку по оси X
xA = np.linspace(sortedA.min(), sortedA.max(), 300)

# 4) Вычисляем оценку плотности на этой сетке
pdfA = kdeA(xA)

# 5) Строим сглаженный CDF (интеграл от плотности)
#    Приблизим интеграл через накопленную сумму (trapz или cumsum):
dxA = xA[1] - xA[0]
cdfA = np.cumsum(pdfA) * dxA  # интегрируем плотность по сетке
# Нормируем, чтобы последний элемент был ≈ 1
cdfA = cdfA / cdfA[-1]

# 6) Кумулята (проценты)
cumA = cdfA * 100

# 7) Огива (накопленная абсолютная частота)
ogivaA = cdfA * nA

# 8) Строим три графика рядом
plt.figure(figsize=(12, 4))

# (a) Сглаженная ЭФР
plt.subplot(1, 3, 1)
plt.title("Функция распределения (CDF)")
plt.plot(xA, cdfA, color='blue', label='ЭФР')
plt.xlabel("X")
plt.ylabel("F(X)")
plt.grid(True)
plt.legend()

# (b) Кумулята
plt.subplot(1, 3, 2)
plt.title("Кумулята")
plt.plot(xA, cumA, color='red', label='Кумулята')
plt.xlabel("X")
plt.ylabel("Процент, %")
plt.grid(True)
plt.legend()

# (c) Огива
plt.subplot(1, 3, 3)
plt.title("Огива")
plt.plot(xA, ogivaA, color='green', label='Огива')
plt.xlabel("X")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
