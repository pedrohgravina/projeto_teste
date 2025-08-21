import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import f_oneway
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import friedmanchisquare
from scipy.stats import kruskal

def tabela_distribuicao_frequencias(dataframe, coluna, coluna_frequencia=False):
    df_estatistica = pd.DataFrame()

    if coluna_frequencia:
        df_estatistica["frequencia"] = dataframe[coluna]
        df_estatistica["frequencia_relativa"] = df_estatistica["frequencia"] / df_estatistica["frequencia"].sum()
    else:
        df_estatistica["frequencia"] = dataframe[coluna].value_counts().sort_index()
        df_estatistica["frequencia_relativa"] = dataframe[coluna].value_counts(normalize=True).sort_index()

    df_estatistica["frequencia_acumulada"] = df_estatistica["frequencia"].cumsum()
    df_estatistica["frequencia_relativa_acumulada"] = df_estatistica["frequencia_relativa"].cumsum()

    return df_estatistica

def composicao_histograma_boxplot(dataframa, coluna, intervalos="auto"):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios": (0.15, 0.85), "hspace": 0.02}, sharex=True)
    
    sns.boxplot(
        data=dataframa, 
        x=coluna, 
        ax=ax1,
        showmeans=True,
        meanline=True,
        meanprops={"color": "C1", "linewidth": 1.5, "linestyle":"--"},
        medianprops={"color": "C3", "linewidth": 1.5, "linestyle":"--"}
    )
    sns.histplot(data=dataframa, x=coluna, kde=True, ax=ax2, bins=intervalos)
    
    for ax in (ax1, ax2):
        ax.grid(True, linestyle="--", color="gray", alpha=0.3)
        ax.set_axisbelow(True)
    
    ax2.axvline(dataframa[coluna].mean(), color="C1", linestyle="--", label="Média")
    ax2.axvline(dataframa[coluna].mode()[0], color="C2", linestyle="--", label="Moda")
    ax2.axvline(dataframa[coluna].median(), color="C3", linestyle="--", label="Mediana")
    # Moda tem o mesmo valor da mediana, por isso sumiu
    
    plt.legend()
    plt.show()

def analise_shapiro(dataframe, alfa=0.05):
    print("Teste de Shapiro-Wilk")

    for coluna in dataframe.columns:
        estatistica, p_value = shapiro(dataframe[coluna])
        print(f"{estatistica:.3f}")
        if p_value > alfa:
            print(f"A coluna: '{coluna}' segue uma distribuição normal. Valor de p: ({p_value:.3f})")
        else:
            print(f"A coluna: '{coluna}' não segue uma distribuição normal. Valor de p: ({p_value:.3f})")

def analise_levene(dataframe, alfa=0.05, centro="mean"):
    print("Teste de Levene")

    estatistica, p_value = levene(*[dataframe[coluna] for coluna in dataframe.columns], center=centro, nan_policy="omit")

    print(f"{estatistica:.3f}")
    if p_value > alfa:
        print(f"Variâncias iguais (São homogêneas). p_value: ({p_value:.3f})")
    else:
        print(f"Ao menos uma variância é diferente (Não são homogêneas). p_value: ({p_value:.3f})")

def analises_shapiro_levene(dataframe, alfa=0.05, centro="mean"):
    analise_shapiro(dataframe, alfa)

    print()

    analise_levene(dataframe, alfa, centro)

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

def analise_ttest_rel(dataframe, alfa=0.05, variancias_iguais=True, alternativa="two-sided"):
    print("Teste t de Student")
    estatistica_ttest, valor_p_ttest = ttest_rel(
        *[dataframe[coluna] for coluna in dataframe.columns],
        alternative=alternativa,
        nan_policy="omit"
    )
    print(f"{estatistica_ttest:.3f}")
    if valor_p_ttest > alfa:
        print(f"Não rejeita a hipótese nula. p_value: ({valor_p_ttest:.3f})")
    else:
        print(f"Rejeita a hipótese nula. p_value: ({valor_p_ttest:.3f})")

def analise_anova_one_way(dataframe, alfa=0.05):
    print("Teste de ANOVA One Way")
    estatistica_f, valor_p_f = f_oneway(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit"
    )
    print(f"{estatistica_f:.3f}")
    if valor_p_f > alfa:
        print(f"Não rejeita a hipótese nula. p_value: ({valor_p_f:.3f})")
    else:
        print(f"Rejeita a hipótese nula. p_value: ({valor_p_f:.3f})")

def analise_wilcoxon(dataframe, alfa=0.05, alternativa="two-sided"):
    print("Teste de Wilcoxon")
    estatistica_wilcoxon, valor_p_wilcoxon = wilcoxon(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
        alternative=alternativa
    )
    print(f"Estatística Wilcoxon: {estatistica_wilcoxon:.3f}")
    if valor_p_wilcoxon > alfa:
        print(f"Não rejeita a hipótese nula. p_value: ({valor_p_wilcoxon:.3f})")
    else:
        print(f"Rejeita a hipótese nula. p_value: ({valor_p_wilcoxon:.3f})")

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

def analise_friedmanchisquare(dataframe, alfa=0.05):
    print("Teste de Friedman")
    estatistica_friedmanchisquare, valor_p_friedmanchisquare = friedmanchisquare(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
    )
    print(f"Estatística Friedman: {estatistica_friedmanchisquare:.3f}")
    if valor_p_friedmanchisquare > alfa:
        print(f"Não rejeita a hipótese nula. p_value: ({valor_p_friedmanchisquare:.3f})")
    else:
        print(f"Rejeita a hipótese nula. p_value: ({valor_p_friedmanchisquare:.3f})")

def analise_kruskal(dataframe, alfa=0.05):
    print("Teste de Kruskal")
    estatistica_kruskal, valor_p_kruskal = kruskal(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
    )
    print(f"Estatística Kruskal: {estatistica_kruskal:.3f}")
    if valor_p_kruskal > alfa:
        print(f"Não rejeita a hipótese nula. p_value: ({valor_p_kruskal:.3f})")
    else:
        print(f"Rejeita a hipótese nula. p_value: ({valor_p_kruskal:.3f})")

def remove_outliers(dados, largura_bigodes=1.5):
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1
    return dados[(dados >= q1 - largura_bigodes * iqr) & (dados <= q3 + largura_bigodes * iqr)]
    
