# Importação de bibliotecas
import pandas as pd
from pycaret.classification import *

# 1) Carrega os dados
data = pd.read_csv('doc/mushrooms.csv', sep=',')
target_col = 'class'

# 2) Setup do PyCaret
s = setup(
    data=data,
    target=target_col,
    session_id=1,
    # Ajustes: em datasets 100% categóricos, pode ser melhor desativar
    remove_multicollinearity=False,  # evita descartar dummies à toa
    feature_selection=False,         # desative no começo (pode ativar depois)
    low_variance_threshold=0.01
)

# 3) Compara modelos e salva leaderboard
best = compare_models(sort='Accuracy')
leaderboard = pull()
leaderboard.to_csv('doc/leaderboard_models.csv', index=False)  # ranking de todos os modelos

# 4) Avalia o melhor modelo no holdout e salva métricas
predict_model(best)
holdout_results = pull()  # tabela de métricas do holdout
holdout_results.to_csv('doc/holdout_metrics.csv', index=False)

# 5) Importância de features do melhor modelo
plot_model(best, plot='feature')
feature_importance = pull()
feature_importance.to_csv('doc/feature_importance.csv', index=False)

# 6) Lista final de features utilizadas no pipeline
selected_feature_names = get_config('X_train').columns.tolist()
pd.Series(selected_feature_names, name='features').to_csv('doc/selected_features.csv', index=False)

# 7) Salva o modelo final para reuso
final_best = finalize_model(best)
save_model(final_best, 'doc/best_model')