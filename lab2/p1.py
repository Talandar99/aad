#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# biblioteka pandas posłuży nam do wczytania zbioru danych do obiektu DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# SPRAWDZ SCIEZKE

df = pd.read_csv("iris.data.wn", header=None)
# wyświetlimy informacje o liczbie wczytanych wierszy
print('liczba rekordów we wczytamym pliku wynosi:', len(df))

# sprawdźmy jakie mamy wogóle kategorie
Z = df.iloc[0:150, 4].values
print('liczba etykiet we wczytamym zbiorze wynosi:', np.unique(Z))

# a teraz informacje statystyczne (opisowe) danych
print(df.describe())


# wyświetlimy dane które wczytaliśmy, ale 4 ostatnie wiersze
print(df.tail(4))

# tu sobie narysujemy wykres z cześci danych
# wybiera gatunki setosa i versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# wydobywa cechy: długość działki i długość płatka
X = df.iloc[0:100, [0, 2]].values

# rysuje wykres
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')

# rysuje wykres
plt.show()


# Sprawdźmy czy wystepują jakieś brakujące dane - tu jes wyświetlimy
print('Tyle wierszy ma null: ', df.isnull().sum().sum())
# a teraz wyswietlimy te wiersze z NaN
selected_rows = df[df[0].isnull() | df[1].isnull() |
                   df[2].isnull() | df[3].isnull()]
print(selected_rows)


# lub alternatywnie ich szukamy i wyświetlimy
print('a teraz pokażmy jak alternatywnie mozemy ich wyszukać')
nan_rows = df.loc[df.isna().any(axis=1)]
# i też ich wyswietlenie
print(nan_rows)


# brakujące dane wypełnimy średnią
# a teraz w petli przechodzimy kolejne kolumny w danych; liczymy średnie po kolumnach i tymi średnimi wypełniamy NaN
for i in range(len(df.columns)-1):
    missing_col = [i]
    for j in missing_col:
        df.loc[df.loc[:, i].isnull(), j] = df.loc[:, j].mean()

# Sprawdźmy czy wystepują jakieś brakujące dane NaN - tu jes wyświetlimy
print('NaN występuje teraz: ', df.isnull().sum().sum(), ' razy')
# i wyswietlamy te wiersze
if df.isnull().sum().sum() > 0:
    selected_rows = df[df[0].isnull() | df[1].isnull() |
                       df[2].isnull() | df[3].isnull()]
    print(selected_rows)


# standaryzacja - ale poniżej zrobiono to tylko dla 1 kolumny o indeksie 0
for i in range(len(df.columns)-1):
    df[i] = (df[i] - df[i].mean()) / df[i].std()


# wyświetlimy dane które zostały poddane standaryzacji
print(df)


# zapis do pliku - ALE SPRAWDZ SCIEZKE
df.to_csv("iris.data.output", index=False, header=False)

# NORMALIZACJA
df_norm = df.copy()

for i in range(len(df_norm.columns) - 1):
    col = df_norm[i].astype(float)
    denom = col.max() - col.min()
    df_norm[i] = (col - col.min()) / denom if denom != 0 else 0.0

df_norm.to_csv("iris.data.normalized.csv", index=False, header=False)


df_std = df.copy()

for i in range(len(df_std.columns) - 1):
    col = df_std[i].astype(float)
    df_std[i] = (col - col.mean()) / col.std()

X = df_std.iloc[:, [0, 1]].values
labels = df_std.iloc[:, 4].values

plt.figure()
for cls in np.unique(labels):
    mask = labels == cls
    plt.scatter(X[mask, 0], X[mask, 1], label=str(cls))

plt.xlabel("Kolumna 1 po standaryzacji (V1)")
plt.ylabel("Kolumna 2 po standaryzacji (V2)")
plt.legend()
plt.savefig("scatter_standardized.png")
plt.close()
