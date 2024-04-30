# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train = pd.read_csv('input/train.csv', index_col='id').reset_index(drop=True)
test = pd.read_csv('input/test.csv').reset_index(drop=True)
sample_submission = pd.read_csv('input/submission_sample.csv')
meta = pd.read_csv('input/metadata.csv')

cat_nom = [x for x in meta.iloc[1:-1, :].loc[(meta.iloc[:,1]=="Qualitativo nominal")].iloc[:,0]]
cat_ord = [x for x in meta.iloc[1:-1, :].loc[(meta.iloc[:,1]=="Qualitativo ordinal")].iloc[:,0]]
num_dis = [x for x in meta.iloc[1:-1, :].loc[(meta.iloc[:,1]=="Quantitativo discreto")].iloc[:,0]]
num_con = [x for x in meta.iloc[1:-1, :].loc[(meta.iloc[:,1]=="Quantitativo continua")].iloc[:,0]]

# Separando os dados em features (X) e alvo (y)
X = train.drop('y', axis=1)
y = train['y']

# Dividindo os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.1, random_state=15)

# Pipeline para pré-processamento dos dados
pipeline_quantitativo = Pipeline([
    ('scaler', StandardScaler())  # Padronizando os dados quantitativos
])

pipeline_qualitativo = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Codificando os dados qualitativos
])

transformador = ColumnTransformer([
    ('quantitativo', pipeline_quantitativo, num_con+num_dis),  # Aplica o pipeline aos dados quantitativos
    ('qualitativo', pipeline_qualitativo, cat_ord+cat_nom)  # Aplica o pipeline aos dados qualitativos
], remainder='passthrough')

# Pipeline completo com pré-processamento e modelo
pipeline_completo = Pipeline([
    ('preprocessamento', transformador),
    ('classificador', GradientBoostingClassifier(random_state=15))
])

# Treinando o modelo
pipeline_completo.fit(X_treino, y_treino)

# Fazendo previsões
previsoes = pipeline_completo.predict(X_teste)

# Calculando a precisão do modelo
precisao = accuracy_score(y_teste, previsoes)
print("Precisão do modelo:", precisao)


# Fazer previsões de classe
previsoes_classes = pipeline_completo.predict(test)

# Fazer previsões de probabilidades
probabilidades = pipeline_completo.predict_proba(test)

# Imprimir as previsões de classe e probabilidades para as primeiras amostras de teste
for i in range(len(test)):
    print(f"Linha {i+1}:")
    print(f"   - Classe prevista: {previsoes_classes[i]}")
    print(f"   - Probabilidade de 0: {probabilidades[i][0]:.2f}")
    print(f"   - Probabilidade de 1: {probabilidades[i][1]:.2f}")