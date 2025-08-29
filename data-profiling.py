import pandas as pd
from ydata_profiling import ProfileReport

df=pd.read_csv(f"doc/mushrooms.csv")
df.head()

profile = ProfileReport(df, title="Pandas Profiling Report Mushroom Classification")

# Caminho onde o relatório será salvo
report_path = f'doc/pandas_profiling_mushroom_classification.html'

# Salve o relatório como um arquivo HTML
profile.to_file(report_path)