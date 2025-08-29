# Inicializa o ambiente PyCaret para classificação:
import pandas as pd
from pycaret.classification import *
from pycaret.classification import get_config

# 1) Carrega os dados em um DataFrame pandas:
data = pd.read_csv('doc/mushrooms.csv', sep=',')
target_col = 'class'  # define o nome da coluna alvo (variável resposta) exatamente como aparece no DataFrame

# 2) Setup do PyCaret com remoção de multicolinearidade e seleção automática das features (variáveis) mais importantes:
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

# 3) Compara modelos e guarda o leaderboard:
best = compare_models()
leaderboard = pull()  # ranking dos modelos

# 4) Importância de features do melhor modelo:
plot_model(best, plot='feature')
feature_importance = pull()  # tabela de importâncias

# 5) Lista final de features que entraram no modelo (após todo o pipeline):
selected_feature_names = get_config('X_train').columns.tolist()
print('Total de features utilizadas:', len(selected_feature_names))
print(selected_feature_names[:25], '...')

# 6) Salva as features selecionadas em um CSV para referência futura:
pd.Series(selected_feature_names, name='features').to_csv(
    'doc/selected_features.csv', index=False
)