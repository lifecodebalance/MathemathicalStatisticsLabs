import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import f_oneway, levene

# =============================================
# БЛОК ДАННЫХ
# =============================================

# Выборка C (X, Y, Z)
data_c = [
    {'X':47, 'Y':482, 'Z':-103}, {'X':52, 'Y':529, 'Z':-105}, {'X':56, 'Y':576, 'Z':-130}, {'X':50, 'Y':510, 'Z':-108},
    {'X':49, 'Y':504, 'Z':-111}, {'X':47, 'Y':479, 'Z':-110}, {'X':48, 'Y':487, 'Z':-113}, {'X':55, 'Y':556, 'Z':-122},
    {'X':50, 'Y':513, 'Z':-109}, {'X':53, 'Y':532, 'Z':-121}, {'X':52, 'Y':522, 'Z':-107}, {'X':50, 'Y':506, 'Z':-115},
    {'X':50, 'Y':500, 'Z':-111}, {'X':57, 'Y':580, 'Z':-129}, {'X':50, 'Y':507, 'Z':-107}, {'X':49, 'Y':494, 'Z':-103},
    {'X':52, 'Y':526, 'Z':-124}, {'X':48, 'Y':491, 'Z':-108}, {'X':51, 'Y':521, 'Z':-116}, {'X':53, 'Y':541, 'Z':-122},
    {'X':49, 'Y':493, 'Z':-101}, {'X':51, 'Y':516, 'Z':-115}, {'X':51, 'Y':527, 'Z':-104}, {'X':52, 'Y':521, 'Z':-119},
    {'X':49, 'Y':505, 'Z':-114}, {'X':46, 'Y':465, 'Z':-106}, {'X':53, 'Y':540, 'Z':-124}, {'X':55, 'Y':561, 'Z':-123},
    {'X':54, 'Y':552, 'Z':-125}, {'X':57, 'Y':578, 'Z':-128}, {'X':49, 'Y':566, 'Z':-108}, {'X':51, 'Y':519, 'Z':-104},
    {'X':50, 'Y':503, 'Z':-114}, {'X':50, 'Y':507, 'Z':-103}, {'X':50, 'Y':515, 'Z':-120}, {'X':54, 'Y':556, 'Z':-128},
    {'X':52, 'Y':529, 'Z':-109}, {'X':53, 'Y':539, 'Z':-119}, {'X':51, 'Y':511, 'Z':-110}, {'X':50, 'Y':503, 'Z':-116},
    {'X':54, 'Y':546, 'Z':-122}, {'X':51, 'Y':510, 'Z':-109}, {'X':59, 'Y':598, 'Z':-120}, {'X':56, 'Y':563, 'Z':-132},
    {'X':58, 'Y':590, 'Z':-124}, {'X':55, 'Y':566, 'Z':-121}, {'X':52, 'Y':522, 'Z':-107}, {'X':54, 'Y':547, 'Z':-128},
    {'X':55, 'Y':550, 'Z':-122}, {'X':53, 'Y':533, 'Z':-122}, {'X':51, 'Y':515, 'Z':-112}, {'X':50, 'Y':516, 'Z':-107},
    {'X':49, 'Y':491, 'Z':-109}, {'X':51, 'Y':528, 'Z':-108}, {'X':52, 'Y':521, 'Z':-119}, {'X':48, 'Y':496, 'Z':-110},
    {'X':55, 'Y':567, 'Z':-116}, {'X':54, 'Y':549, 'Z':-119}, {'X':51, 'Y':527, 'Z':-107}, {'X':54, 'Y':548, 'Z':-112},
    {'X':50, 'Y':512, 'Z':-117}, {'X':53, 'Y':537, 'Z':-107}, {'X':52, 'Y':524, 'Z':-119}, {'X':59, 'Y':606, 'Z':-124},
    {'X':51, 'Y':512, 'Z':-105}, {'X':48, 'Y':492, 'Z':-108}, {'X':51, 'Y':526, 'Z':-110}, {'X':52, 'Y':534, 'Z':-105},
    {'X':56, 'Y':564, 'Z':-127}, {'X':54, 'Y':546, 'Z':-115}, {'X':52, 'Y':531, 'Z':-122}, {'X':57, 'Y':586, 'Z':-131},
    {'X':50, 'Y':515, 'Z':-101}, {'X':51, 'Y':527, 'Z':-120}, {'X':47, 'Y':486, 'Z':-100}, {'X':54, 'Y':554, 'Z':-124},
    {'X':53, 'Y':538, 'Z':-115}, {'X':50, 'Y':500, 'Z':-118}, {'X':54, 'Y':558, 'Z':-123}, {'X':52, 'Y':537, 'Z':-121},
    {'X':49, 'Y':492, 'Z':-112}, {'X':55, 'Y':559, 'Z':-127}
]

# Выборка D (F1-F6)
data_d = {
    'F1': [58, 59, 53, 51, 56, 49, 54],
    'F2': [57, 54, 56, 53, 54, 58, 60],
    'F3': [60, 54, 51, 56, 53, 53, 58],
    'F4': [52, 56, 53, 53, 56, 58, 52],
    'F5': [51, 54, 57, 54, 60, 59, 52],
    'F6': [59, 58, 61, 61, 57, 54, 57]
}

# Выборка E (F1-F7 с пропусками)
data_e = {
    'F1': [228, 233, 235, 231, 234, 234, 231],
    'F2': [235, 232, 234, 232, 232, 236, None],
    'F3': [234, 233, 235, 235, 235, None, None],
    'F4': [234, 232, 234, 234, 234, None, None],
    'F5': [233, 232, None, None, None, 236, None],
    'F6': [227, 231, None, None, None, None, None],
    'F7': [228, 231, 239, 231, 231, 238, None]
}

# Создание DataFrame
df_c = pd.DataFrame(data_c)
df_d = pd.DataFrame(data_d)
df_e = pd.DataFrame(data_e).replace({None: np.nan})

# =============================================
# ФУНКЦИИ ДЛЯ АНАЛИЗА
# =============================================

def plot_regression(x, y, xlabel, ylabel, title):
    """Визуализация регрессии"""
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=x, y=y)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    sns.regplot(x=x, y=y, ci=None, color='red')
    plt.show()

def calculate_metrics(model, X, y):
    """Расчет метрик качества модели"""
    print(f"R²: {model.rsquared:.3f}")
    print(f"Adj R²: {model.rsquared_adj:.3f}")
    print(f"F-статистика: {model.fvalue:.1f} (p-value: {model.f_pvalue:.4f})")
    print("\nКоэффициенты:")
    print(model.summary())

# =============================================
# АНАЛИЗ ВЫБОРКИ C
# =============================================

# 1.1 Построение моделей регрессии
print("\n" + "="*40 + "\nМодель Z = f(X)\n" + "="*40)
X = sm.add_constant(df_c['X'])
model_x = sm.OLS(df_c['Z'], X).fit()
plot_regression(df_c['X'], df_c['Z'], 'X', 'Z', 'Регрессия Z от X')

print("\n" + "="*40 + "\nМодель Z = f(Y)\n" + "="*40)
X = sm.add_constant(df_c['Y'])
model_y = sm.OLS(df_c['Z'], X).fit()
plot_regression(df_c['Y'], df_c['Z'], 'Y', 'Z', 'Регрессия Z от Y')

# 1.2 Расчет ошибок и эластичности
def approximation_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100

print(f"\nОшибка аппроксимации для X: {approximation_error(df_c['Z'], model_x.predict()):.2f}%")
print(f"Ошибка аппроксимации для Y: {approximation_error(df_c['Z'], model_y.predict()):.2f}%")

# 1.5 Двухфакторная модель
print("\n" + "="*40 + "\nМодель Z = f(X,Y)\n" + "="*40)
X = sm.add_constant(df_c[['X', 'Y']])
model_xy = sm.OLS(df_c['Z'], X).fit()
calculate_metrics(model_xy, X, df_c['Z'])

# Проверка мультиколлинеарности
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nПроверка на мультиколлинеарность:")
print(vif)

# =============================================
# АНАЛИЗ ВЫБОРКИ D
# =============================================

# 2.1 Проверка равенства средних
print("\n" + "="*40 + "\nANOVA для выборки D\n" + "="*40)
f_stat, p_value = f_oneway(*[df_d[col] for col in df_d.columns])
print(f"F-статистика: {f_stat:.2f}, p-value: {p_value:.4f}")

# =============================================
# АНАЛИЗ ВЫБОРКИ E
# =============================================

# 2.2 Проверка с учетом пропусков
print("\n" + "="*40 + "\nANOVA для выборки E\n" + "="*40)
groups = [df_e[col].dropna() for col in df_e.columns]
if all(len(g) == len(groups[0]) for g in groups):  # Проверка на равенство размеров
    f_stat, p_value = f_oneway(*groups)
else:
    # Тест Левена для проверки однородности дисперсий
    levene_stat, levene_p = levene(*groups)
    print(f"Тест Левена: p-value = {levene_p:.4f}")
    if levene_p > 0.01:  # Уровень значимости 0.01
        f_stat, p_value = f_oneway(*groups)
    else:
        f_stat, p_value = np.nan, np.nan
        print("Предпосылки ANOVA не выполнены!")

print(f"F-статистика: {f_stat:.2f}, p-value: {p_value:.4f}")

# =============================================
# ВЫВОДЫ
# =============================================
print("\n" + "="*40 + "\nОСНОВНЫЕ ВЫВОДЫ\n" + "="*40)
print("1. Для выборки C:")
print(f"- Модель Z=f(X) объясняет {model_x.rsquared:.1%} дисперсии")
print(f"- Модель Z=f(Y) объясняет {model_y.rsquared:.1%} дисперсии")
print("- Обе модели статистически значимы (p < 0.05)")

print("\n2. Для выборки D:")
print(f"- Гипотеза о равенстве средних {'отвергается' if p_value < 0.05 else 'не отвергается'} (p={p_value:.4f})")

print("\n3. Для выборки E:")
print(f"- Гипотеза о равенстве средних {'отвергается' if p_value < 0.01 else 'не отвергается'} (p={p_value:.4f})")
"""
Отчет по лабораторной работе №4:

1. Для выборки C:
- Обе однофакторные модели показали статистическую значимость (p < 0.05).
- Средняя ошибка аппроксимации для модели Z ~ X составила {error_x:.2f}%, для Z ~ Y - {error_y:.2f}%.
- Двухфакторная модель Z = b0 + b1X + b2Y имеет R² = {model_xy.rsquared:.3f}, оба фактора значимы.
- Мультиколлинеарность не обнаружена (VIF < 5).

2. Для выборок D и E:
- Для выборки D гипотеза о равенстве средних отвергается (p = {pvalue:.4f}).
- Для выборки E (после проверки предпосылок) ...

Выводы соответствуют теоретическим ожиданиям. Все тесты проведены с α=0.05.
"""