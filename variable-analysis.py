# Inicializa o ambiente PyCaret para classificação:
import pandas as pd
from pycaret.classification import *

# 1) Carrega os dados em um DataFrame pandas:
data = pd.read_csv('doc/mushrooms.csv', sep=',')
target_col = 'class' # define o nome da coluna alvo (variável resposta) exatamente como aparece no DataFrame

# 2) Remove linhas com valores NaN da coluna indicada e reorganiza os índices em ordem crescente, descartando os antigos:
# data = data.dropna(subset=['']).reset_index(drop=True)

# Opcional: verifica rapidamente
# print(data.head())
# print(data.columns)

# 3) Setup do PyCaret com remoção de multicolinearidade e seleção automática das features (variáveis) mais importantes:
s = setup(
    data=data,
    target=target_col,
    low_variance_threshold=0.01,      # 0 = remove só constantes; use 0.01 para quase-constantes
    remove_multicollinearity=True,    # remove variáveis altamente correlacionadas
    multicollinearity_threshold=0.9,  # corte de correlação (0.9 é comum)
    feature_selection=True,           # "classic" por padrão no pycaret 3
    n_features_to_select=0.8,         # mantém ~80% das features (pode usar um inteiro também)
    session_id=1                      # reprodutibilidade
)

# 4) Treina um modelo para obter importância de features (calculada em relação a um modelo específico):
best = compare_models()  # escolhe automaticamente o melhor entre vários (é também possível escolher diretamente um algoritmo)

# 5) Captura a importância de features:
plot_model(best, plot='feature')   # gera o gráfico
fi = pull()                        # 'puxa' a tabela exibida (DataFrame com importâncias)

# 6) Visualiza as features selecionadas:
selected_features = pull()
print(selected_features['Description'].values)

# Visualize/filtre como preferir
# print(fi.head(20))