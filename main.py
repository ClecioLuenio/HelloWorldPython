import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1) Carregar dataset (ajuste o nome do arquivo caso diferente)
df = pd.read_csv("patients.csv")

# 2) Explorar os dados
print("Primeiras linhas do dataset:")
print(df.head())

print("\nDistribuição da variável alvo (Risk_Level):")
print(df["Risk_Level"].value_counts())

# 3) Pré-processamento
# Remover colunas que não ajudam no modelo
df = df.drop(columns=["Patient_ID"])

# Transformar colunas categóricas em numéricas
le = LabelEncoder()
df["Consciousness"] = le.fit_transform(df["Consciousness"])
df["Risk_Level"] = le.fit_transform(df["Risk_Level"])  # variável alvo

# 4) Definir variáveis independentes (X) e alvo (y)
X = df.drop(columns=["Risk_Level"])
y = df["Risk_Level"]

# 5) Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6) Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7) Avaliar modelo
y_pred = model.predict(X_test)
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 8) Visualizar importância das variáveis
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Importância das Variáveis")
plt.show()
