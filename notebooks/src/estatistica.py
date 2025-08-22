import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, ttest_ind, mannwhitneyu

def analise_levene(dataframe, alfa=0.05, centro="mean"):
    print("Teste de Levene")

    estatistica, p_value = levene(*[dataframe[coluna] for coluna in dataframe.columns], center=centro, nan_policy="omit")

    print(f"{estatistica:.3f}")
    if p_value > alfa:
        print(f"Variâncias iguais (São homogêneas). p_value: ({p_value:.3f})")
    else:
        print(f"Ao menos uma variância é diferente (Não são homogêneas). p_value: ({p_value:.3f})")

def analise_ttest_ind(dataframe, alfa=0.05, variancias_iguais=True, alternativa="two-sided"):
    print("Teste t de Student")
    estatistica_ttest, valor_p_ttest = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns],
        equal_var=variancias_iguais,
        alternative=alternativa,
        nan_policy="omit"
    )
    print(f"{estatistica_ttest:.3f}")
    if valor_p_ttest > alfa:
        print(f"Não rejeita a hipótese nula. p_value: ({valor_p_ttest:.3f})")
    else:
        print(f"Rejeita a hipótese nula. p_value: ({valor_p_ttest:.3f})")

def analise_mannwhitneyu(dataframe, alfa=0.05, alternativa="two-sided"):
    print("Teste de Mann-Whitney")
    estatistica_mannwhitneyu, valor_p_mannwhitneyu = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
        alternative=alternativa
    )
    print(f"Estatística Mann-Whitney: {estatistica_mannwhitneyu:.3f}")
    if valor_p_mannwhitneyu > alfa:
        print(f"Não rejeita a hipótese nula. p_value: ({valor_p_mannwhitneyu:.3f})")
    else:
        print(f"Rejeita a hipótese nula. p_value: ({valor_p_mannwhitneyu:.3f})")

def remove_outliers(dados, largura_bigodes=1.5):
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1
    return dados[(dados >= q1 - largura_bigodes * iqr) & (dados <= q3 + largura_bigodes * iqr)]
    
