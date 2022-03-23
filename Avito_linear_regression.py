# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import statsmodels.api as sm # для построения линейной регрессии
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # нормализация значений
from sklearn.model_selection import train_test_split # сплит на тестовую и тренировочную выборки
from sklearn.linear_model import LinearRegression # также для построения линейной регрессии
sns.set() # используем стили seaborn по умолчанию

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# импортируем спарсенные с Авито данные
df = pd.read_csv('/Users/dmitrii/Documents/PyCharmProjects/Avito_parcing/avito_cars.csv')

# Всегда полезно открыть в том числе и сам файл и бегло просмотреть данные
# Я обнаружил, что некоторые строки были занесены не корректно
df['Состояние'].value_counts()

# как видим у нас 11 битых машин и 28 с некорретными значениями, можем просто удалить их из дата-сета
df_no_crash = df[df['Состояние'] == 'не битый']

# также заменяем владельцев 4+ на просто 4
df_no_crash = df_no_crash.replace({'Владельцев по ПТС': '4+'}, '4')

# удаляем ненужные нам столбцы
del df_no_crash['Unnamed: 0']
del df_no_crash['Состояние']

# сбрасываем индексы
df_no_crash = df_no_crash.reset_index(drop=True)

# преобразуем в числовые типы данных некоторые столбцы
df_no_crash['Год выпуска'] = df_no_crash['Год выпуска'].astype('int')
df_no_crash['Пробег'] = df_no_crash['Пробег'].astype('int')
df_no_crash['Владельцев по ПТС'] = df_no_crash['Владельцев по ПТС'].astype('int')

# год выпуска
# sns.distplot(df_no_crash['Год выпуска'])
q = df_no_crash['Год выпуска'].quantile(0.01)
df_clean1 = df_no_crash[df_no_crash['Год выпуска']>q]
# sns.distplot(df_clean1['Год выпуска'])

# пробег
# sns.distplot(df_clean1['Пробег'])
q = df_clean1['Пробег'].quantile(0.99)
df_clean2 = df_clean1[df_clean1['Пробег']<q]
# sns.distplot(df_clean2['Пробег'])

# владельцы по ПТС
# sns.distplot(df_clean2['Владельцев по ПТС'])

# цена автомобиля
# sns.distplot(df_clean2['Цена'])
q = df_clean2['Цена'].quantile(0.99)
df_cleaned = df_clean2[df_clean2['Цена']<q]
# sns.distplot(df_cleaned['Цена'])


y = df_cleaned['Цена']
x1 = df_cleaned[['Год выпуска', 'Пробег', 'Владельцев по ПТС']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> сделать общей ординату y=price
# ax1.scatter(df_cleaned['Год выпуска'],df_cleaned['Цена'])
# ax1.set_title('Цена и год выпуска')
# ax2.scatter(df_cleaned['Владельцев по ПТС'],df_cleaned['Цена'])
# ax2.set_title('Цена и кол-во владельцев')
# ax3.scatter(df_cleaned['Пробег'],df_cleaned['Цена'])
# ax3.set_title('Цена и пробег')

df_cleaned['Цена_лог'] = np.log(df_cleaned['Цена'])

# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> сделать общей ординату y=price
# ax1.scatter(df_cleaned['Год выпуска'],df_cleaned['Цена_лог'])
# ax1.set_title('Цена и год выпуска')
# ax2.scatter(df_cleaned['Владельцев по ПТС'],df_cleaned['Цена_лог'])
# ax2.set_title('Цена и кол-во владельцев')
# ax3.scatter(df_cleaned['Пробег'],df_cleaned['Цена_лог'])
# ax3.set_title('Цена и пробег')

# оценка фактора инфляции дисперсии
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df_cleaned[['Год выпуска','Владельцев по ПТС','Пробег']]
vif = pd.DataFrame()

vif["ФИД"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Переменные"] = variables.columns

data_with_dummies = pd.get_dummies(df_cleaned, drop_first=True)
data_with_dummies.columns.values

# убираем старую цену, она нам больше не нужна
data_with_dummies_clean = data_with_dummies.drop(['Цена'],axis=1)

data_with_dummies = pd.get_dummies(df_cleaned, drop_first=True)
data_with_dummies.columns.values

# объявляем независимые и зависимую перменную
targets = data_with_dummies_clean['Цена_лог']
inputs = data_with_dummies_clean.drop(['Цена_лог'],axis=1)

# нормализируем наши значения
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# делаем сплит из тренировочной выборки и тестовой, разбиваем стандартно 80/20
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

reg = LinearRegression()
reg.fit(x_train,y_train)

y_hat = reg.predict(x_train)
# plt.scatter(y_train, y_hat)
# plt.xlabel('Targets (y_train)',size=18)
# plt.ylabel('Predictions (y_hat)',size=18)
# plt.xlim(11,15)
# plt.ylim(11,15)
# plt.show()

# sns.distplot(y_train - y_hat)
# plt.title("Функция плотности вероятности остатков", size=18)
# plt.show()

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Переменные'])
reg_summary['Веса'] = reg.coef_

y_hat_test = reg.predict(x_test)
# plt.scatter(y_test, y_hat_test, alpha=0.2)
# plt.xlabel('Targets (y_test)',size=18)
# plt.ylabel('Predictions (y_hat_test)',size=18)
# plt.xlim(12,15)
# plt.ylim(12,15)
# plt.show()

x = sm.add_constant(inputs)
results = sm.OLS(targets,x).fit()
results.summary()

y_test = y_test.reset_index(drop=True) # не забываем дропнуть индексы, потому что наш дата сет был "перемешан"
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Предсказанное значение']) # np.exp отвечает за преобразование логарифма
df_pf['Целевое значение'] = np.exp(y_test)
df_pf['Разница'] = df_pf['Целевое значение'] - df_pf['Предсказанное значение']
df_pf['Разница в %'] = np.absolute(df_pf['Разница']/df_pf['Целевое значение']*100)

polo = -195.7369 + 0.1043*2015 + 62000*(-0.00000089) - 1.1371
print(np.exp(polo))