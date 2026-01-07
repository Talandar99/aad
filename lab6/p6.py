#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from kneed import KneeLocator
# zaimportowano dla potrzeb oreślenia punktu łokcia

# from yellowbrick.cluster import SilhouetteVisualizer


# SPRAWDZ SCIEZKE
df = pd.read_csv('clu_dane_9.csv', header=0)


# Sprawdzenie czy sa jakieś null lub NaN
print('Tyle wierszy ma null: ', df.isnull().sum().sum())

# standaryzacja danych
scaler = StandardScaler()
std_df = scaler.fit_transform(df)

# teraz mamy przygotowane dane

# A tu uruchomimy kmeans 9 razy (dla k od 2 do 10), zapiszemy wewnątrzklastrowe sumy kwadratów (sse) dla każdego z klastra
# i potem przedstawimy te wartości na wykresie
# dodatkowo pliczymy wartośc wspłczynnika profilu (tu możesz też poczytać o tym wskaźniku:
#                           https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam )
# obie wielkości wyświetlimy na niezależnych wykresach

sse = []
silhouette_coefficients = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, init='random', n_init=100,
                    max_iter=100, tol=1e-04, random_state=0)
    kmeans.fit(std_df, sample_weight=None)
    sse.append(kmeans.inertia_)
    score = silhouette_score(std_df, kmeans.labels_)
    silhouette_coefficients.append(score)


# wykres SEE
plt.style.use("fivethirtyeight")
plt.plot(range(2, 10), sse)
plt.xticks(range(2, 10))
plt.xlabel("Liczba grup")
plt.ylabel("SSE - wewnątrz klastrowa suma kwadratow")
plt.grid
plt.show()

# wykres silhouette
print(silhouette_coefficients)
plt.style.use("fivethirtyeight")
plt.plot(range(2, 10), silhouette_coefficients)
plt.xticks(range(2, 10))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

# Polecenie nr 1:
# Czy potafiłbyś umieścić sse oraz score na jednym wykresie, prezentując jedną wielkość na osi pomocniczej?
# jeśli tak to zmodyfikuje skrytp, opisz też co zrobiłeś


# Polecenie 2
# Zastosujemy teraz algorytm DBSCAN. Opis tego algorytmu znajdziesz m.in. pod adresem
# https://www.reneshbedre.com/blog/dbscan-python.html
# a) wyznacz parametr minPts dla DBSCAN
# b) przygotuj skrypt do automatycznego wyznaczenia parametru Epsilon w algorytmie DBSCAN, wykreśl też wykres
# pozwalający odczytać tę wartość, opisz działanie skrytpu
# c) dokonaj grupowanie wczytanych danych używając algorytmu DBSCAN, wyświetl informację o liczbie grup wskazanych z użyciem
# tego algorytmu, wyznacz dla uzyskanego pogrupowania danych wartość współczynnika profilu. Czy wynik różni sie od wyniku otrzymanego
# z wykorzsystaniem k-means (niech odpowiedź będzie się generowała automatycznie, tj. stosując warunek
