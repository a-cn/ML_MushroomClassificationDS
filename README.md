# 🍄 Mushroom Classification – Data Profiling e Seleção de Features

## 🎯 Objetivo do Projeto
Este projeto tem como objetivo aplicar técnicas de **Machine Learning** para analisar e classificar cogumelos como **comestíveis (edible)** ou **venenosos (poisonous)** com base em suas características físicas, servindo como um estudo prático, com foco em pré-processamento e análise exploratória de dados categóricos.  
O trabalho incluiu desde a análise exploratória e **data profiling** até a **seleção de features** mais relevantes para o modelo.

---

## 📊 Dataset
- **Fonte:** [Mushroom Classification (Kaggle)](https://www.kaggle.com/datasets/uciml/mushroom-classification)  
- **Observações:** 8124 instâncias;  
- **Atributos:** 22 variáveis categóricas + 1 variável resposta (`class`);  
- **Tipo de Problema**: Classificação.  

---

## 🛠️ Tecnologias e Ferramentas
As dependências estão listadas em [`environment.yml`](./environment.yml). Entre as principais:  

| Ferramenta                             | Finalidade                     |
|----------------------------------------|--------------------------------|
| **Python 3.10**                        | Linguagem principal do projeto |
| **pandas**                             | Manipulação de dados |
| **PyCaret 3.3.2**                      | Setup do ambiente de classificação, seleção de features e comparação de modelos (automação de Machine Learning) |
| **scikit-learn**                       | Modelagem |
| **matplotlib / seaborn**               | Visualizações |
| **pandas-profiling / ydata-profiling** | Geração do relatório de Data Profiling |

---

## 📋 Atividades Realizadas
1. **Data Profiling**:  
   - Análise exploratória automatizada para identificar distribuição de variáveis, valores ausentes, cardinalidade e correlações.  

2. **Seleção de Features**:  
   - Remoção de variáveis redundantes ou pouco relevantes;  
   - Avaliação de multicolinearidade e seleção automática das variáveis mais relevantes via PyCaret.  

3. **Treinamento e Comparação de Modelos**:  
   - Vários algoritmos de classificação foram avaliados (árvores de decisão, random forest, regressão logística, entre outros);  
   - O melhor modelo foi identificado com base na métrica **AUC (Área sob a curva ROC)**, priorizando a capacidade do modelo de separar com precisão cogumelos venenosos de comestíveis em diferentes limiares, visando a evitar falsos negativos (erros de classificação).  

4. **Visualização**:  
   - Geração de gráficos de importância das variáveis após o treinamento dos modelos.  

---

## 📈 Features Mais Relevantes
![Feature Importance](./doc/resultados/tentativa_02/Feature_Importance_Plot_wLine.png)  
*Figura: gráfico de importância das variáveis mais relevantes para o modelo.*

---

## 📖 Glossário de Variáveis

### Variável Resposta
- **class**:  
  - `e` = edible (comestível)  
  - `p` = poisonous (venenoso)  

### Preditores
- **cap-shape**: formato do chapéu  
  - `b`=bell, `c`=conical, `x`=convex, `f`=flat, `k`=knobbed, `s`=sunken  

- **cap-surface**: superfície do chapéu  
  - `f`=fibrous, `g`=grooves, `y`=scaly, `s`=smooth  

- **cap-color**: cor do chapéu  
  - `n`=brown, `b`=buff, `c`=cinnamon, `g`=gray, `r`=green, `p`=pink, `u`=purple, `e`=red, `w`=white, `y`=yellow  

- **bruises**: presença de hematomas  
  - `t`=bruises, `f`=no  

- **odor**: odor do cogumelo  
  - `a`=almond, `l`=anise, `c`=creosote, `y`=fishy, `f`=foul, `m`=musty, `n`=none, `p`=pungent, `s`=spicy  

- **gill-attachment**: fixação das lamelas  
  - `a`=attached, `d`=descending, `f`=free, `n`=notched  

- **gill-spacing**: espaçamento das lamelas  
  - `c`=close, `w`=crowded, `d`=distant  

- **gill-size**: tamanho das lamelas  
  - `b`=broad, `n`=narrow  

- **gill-color**: cor das lamelas  
  - `k`=black, `n`=brown, `b`=buff, `h`=chocolate, `g`=gray, `r`=green, `o`=orange, `p`=pink, `u`=purple, `e`=red, `w`=white, `y`=yellow  

- **stalk-shape**: formato do estipe  
  - `e`=enlarging, `t`=tapering  

- **stalk-root**: raiz do estipe  
  - `b`=bulbous, `c`=club, `u`=cup, `e`=equal, `z`=rhizomorphs, `r`=rooted, `?`=missing  

- **stalk-surface-above-ring**: superfície acima do anel  
  - `f`=fibrous, `y`=scaly, `k`=silky, `s`=smooth  

- **stalk-surface-below-ring**: superfície abaixo do anel  
  - `f`=fibrous, `y`=scaly, `k`=silky, `s`=smooth  

- **stalk-color-above-ring**: cor do estipe acima do anel  
  - `n`=brown, `b`=buff, `c`=cinnamon, `g`=gray, `o`=orange, `p`=pink, `e`=red, `w`=white, `y`=yellow  

- **stalk-color-below-ring**: cor do estipe abaixo do anel  
  - `n`=brown, `b`=buff, `c`=cinnamon, `g`=gray, `o`=orange, `p`=pink, `e`=red, `w`=white, `y`=yellow  

- **veil-type**: tipo de véu  
  - `p`=partial, `u`=universal  

- **veil-color**: cor do véu  
  - `n`=brown, `o`=orange, `w`=white, `y`=yellow  

- **ring-number**: número de anéis  
  - `n`=none, `o`=one, `t`=two  

- **ring-type**: tipo de anel  
  - `c`=cobwebby, `e`=evanescent, `f`=flaring, `l`=large, `n`=none, `p`=pendant, `s`=sheathing, `z`=zone  

- **spore-print-color**: cor do esporo  
  - `k`=black, `n`=brown, `b`=buff, `h`=chocolate, `r`=green, `o`=orange, `u`=purple, `w`=white, `y`=yellow  

- **population**: densidade populacional  
  - `a`=abundant, `c`=clustered, `n`=numerous, `s`=scattered, `v`=several, `y`=solitary  

- **habitat**: habitat do cogumelo  
  - `g`=grasses, `l`=leaves, `m`=meadows, `p`=paths, `u`=urban, `w`=waste, `d`=woods  

---

## 📂 Estrutura do Repositório
```plaintext
├── doc/
│   ├── resultados/       # Contém os resultados de execuções
│   └── mushrooms.csv     # Dataset em CSV
├── notebooks/            # Contém os Jupyter Notebooks
├── data-profiling.py     # Código para análise exploratória de dados
├── environment.yml       # Arquivo de dependências
└── README.md
```