# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:19:03 2020

@author: miamo
"""

#%%
import nltk
nltk.download('book')
from nltk.book import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


## A partir de la clase 11:
#%% BI-GRAMAS
md_bigrams = list(bigrams(text1))
fdist = FreqDist(md_bigrams)

fdist.most_common(10) # tambien incluyen caracteres especiales y basura

#%% Filtrado
threshold = 2
filtered_bi = [bigram for bigram in md_bigrams if len(bigram[0]) > threshold and len(bigram[1]) > threshold]
filtered_dist = FrreqDist(filtered_bi)
filtered_dist.plot(20) # imprime grafico de bigramas 

#%% TRI-GRAMAS
from nltk.util import ngrams

md_trigrams = list(ngrams(text1), 3)
fdist = FreqDist(md_trigrams)
fdist.most_common(10)

fdist.plot(20)

#%% Colocaciones
""" Son secuencias de palabras que suelen ocurrir en textos o conversaiones con una frecuencia inusualmente alta
Las colocaciones de una palabra son declaraciones fformales ded donde suele ubicerse tipicamente esa palabra """

# usando md_bigrams anterior <-
filtered_bigrams = [b = for b in md_bigrams if len(b[0]) > threshold and len(b) > threshold]
filtered_bigram_dist = FreqDist(filtered_bi)

filtered_words = [w for w in text1 if len(w) > threshold]
filtered_word_dist = FreqDist(filtered_words)

df = pd.DataFrame(0) # Es una tabla de datos 

df['bi_grams'] = list(set(filtered_bi))		# <- Estas son sus tres columnas
df['word_0'] = df['bi_grams'].apply(lambda x: x[0])
df['word_1'] = df['bi_grams'].apply(lambda x: x[1])
"""
	P(w1, w2) := Probabilidad de que el bi-grama (w1, w2) aparezca en el texto
	P(w1) := Probabilidad de que la palabra w1 aparezca en el texto
	P(w2) := Probabilidad de que la palabra w2 aparezca en el texto
	Pointwise mutual information (PMI)
	Pw1w2 = log ( P(w1, w2) / P(w1)P(w2))
	Los valores de PMI suelen ser megativos (los mas grandes, son cercanos a cero)
"""
df['bi_gram_freq'] = df['bi_grams'].apply(lambda x: filtered_bigram_dist[x])
df['word_0_freq'] = df['word_0'].apply(lambda x: filtered_word_dist[x])
df['word_1_freq'] = df['word_1'].apply(lambda x: filtered_word_dist[x]) # con estas filas se podra calcular el PMI

# Ahora una columna para los PMI's
df['PMI'] = df[['bi_gram_freq', 'word_0_freq', 'word_1_freq']].apply(lambda x: np.log2(x.values[0] / ( x.values[1] * x.values[2] ) )) # El x ahora representa la tripleta de valores

df.sort_values(by = 'PMI', ascending = False)

"""
Para un mejor analisis se debe contemplar la combinacion de las frequencias de las palabras y su PMI en conjunto del bi-grama
"""
df['log(bi_gram_freq)')] = df['bi_gram_freq'].apply[lambda x: np.log2(x)]
# Ahora vamos a visualizar esa relacion
fig = px.scatter(x = df['PMI'].values, y = df['log(bi_gram_freq)'], color = df['PMI'].values+df['log(bi_gram_freq)'], hover_name = df['bi_gram'].values, width = 600, height = 600, labels = {'x': 'PMI', 'y': 'Log(bi_gram_freq)'})
fig.show()
""" El .values en df['PMI'].values convierte los valores a una lista """
from nltk.collocations import *
bigram_measure = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(text1)

# Todo lo que tenga frcuencia mayor a 20
finder.apply_freq_filter(20)
finder.nbest(bigram_measure.pmi, 10) # Esto es algo similar a lo que se puede ver el el grafico aqui arriba

# Descargamos corpus e espanol : https:://mailman.uib.no/public/corpora/2007-October/005448.html
nltk.download('cess_esp')
corpus = nltk.corpus.cess_esp.sents()
flatten_corpus = [w for l in corpus for w in l]
print(flatten_corpus[:50])

finder = BigramCollocationFinder.from_documents(corpus)
finder.apply_freq_filter(10)
finder.nbest(bigram_measure.pmi, 10)
