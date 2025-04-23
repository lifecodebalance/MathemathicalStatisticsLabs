import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Предположим, выборка B большая (просто пример)
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

dataB = np.array(sampleB)
nB = len(dataB)

# 1) Сортируем
sortedB = np.sort(dataB)

# 2) KDE
kdeB = gaussian_kde(sortedB)

# 3) Сетка
xB = np.linspace(sortedB.min(), sortedB.max(), 300)

# 4) PDF
pdfB = kdeB(xB)

# 5) Интегрируем PDF, чтобы получить сглаженный CDF
dxB = xB[1] - xB[0]
cdfB = np.cumsum(pdfB) * dxB
cdfB = cdfB / cdfB[-1]

# 6) Кумулята (в %)
cumB = cdfB * 100

# 7) Огива (накопленная абсолютная частота)
ogivaB = cdfB * nB

# 8) Построим
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Функция распределения (CDF)")
plt.plot(xB, cdfB, color='blue', label='ЭФР')
plt.xlabel("X")
plt.ylabel("F(X)")
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.title("Кумулята")
plt.plot(xB, cumB, color='red', label='Кумулята')
plt.xlabel("X")
plt.ylabel("Процент, %")
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.title("Огива")
plt.plot(xB, ogivaB, color='green', label='Огива')
plt.xlabel("X")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
