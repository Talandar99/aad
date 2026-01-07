#!/usr/bin/env python3

import pandas as pd

import seaborn as sns  # narzędzie do wizualizacji danych
import scipy.cluster.hierarchy as shc  # dendrogram
import matplotlib.pyplot as plt

# SPRAWDZ SCIEZKE!!
df = pd.read_csv('dane_iris.csv')

# wyświetlimy informacje o wczytanych danych
print('liczba rekordów we wczytamym pliku wynosi:', len(df))
print(df.head())
print(df.shape)

num = df.select_dtypes(include="number")
x_col = num.columns[0]
y_col = num.columns[1]

# wykres rozrzutu punktów
sns.relplot(data=num, x=x_col, y=y_col)
plt.savefig("scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# wykres macierzowy - kazdy z kazdym
sns.pairplot(data=num)
plt.savefig("pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

# dendogram
dendrogram = shc.dendrogram(shc.linkage(num, method='ward'))
plt.title('Dendrogram')
plt.xlabel('nr instancji (wiersza w naszych danych)')
plt.ylabel('Euclidean distances')
plt.savefig("dendrogram.png", dpi=150, bbox_inches="tight")
plt.close()
