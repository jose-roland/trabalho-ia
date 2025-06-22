#------------------------------------------------------------------
# ETAPA 1: COLETA, LIMPEZA E EDA (INÍCIO)
#------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb

# --- Configurações Iniciais ---
pd.set_option('future.no_silent_downcasting', True)

# --- Carregamento dos Dados ---
dataframe = pd.read_csv("customerchurn.csv")

# --- Limpeza e Pré-processamento Inicial ---

# Converte TotalCharges para numérico, tratando erros
dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")

# Preenche valores nulos em TotalCharges com 0 (clientes novos sem cobrança total)
dataframe.loc[dataframe["TotalCharges"].isnull(), "TotalCharges"] = 0

# Remove espaços em branco das colunas de texto
for column in dataframe.columns:
    if dataframe[column].dtype == object:
        dataframe[column] = dataframe[column].str.strip()

#------------------------------------------------------------------
# ETAPA 4.2: ENGENHARIA DE ATRIBUTOS (PARTE 1)
#------------------------------------------------------------------
# Criando a feature derivada ANTES do escalonamento para usar os valores originais.
# [cite_start]Isso segue a recomendação do projeto de criar features como "gastos médios". [cite: 15]

# Usamos np.divide para evitar erros de divisão por zero onde 'tenure' é 0
dataframe['gastos_medios'] = np.divide(dataframe['TotalCharges'], dataframe['tenure'],
                                       out=np.zeros_like(dataframe['TotalCharges']), where=dataframe['tenure']!=0)

print("--- Feature Derivada Criada ---")
print(dataframe[['tenure', 'TotalCharges', 'gastos_medios']].head())
print("\n")


#------------------------------------------------------------------
# ETAPA 1: COLETA, LIMPEZA E EDA (CONTINUAÇÃO)
#------------------------------------------------------------------

# --- Transformação de Dados ---

# Codifica manualmente colunas binárias e outras com valores "Yes"/"No"
for column in dataframe.columns:
    if dataframe[column].dtype == object:
        dataframe[column] = dataframe[column].replace({
            "No phone service": 0,
            "No": 0,
            "Yes": 1,
            "0": 0,
            "1": 1
        })

# Converte colunas para tipo booleano para economizar memória e representar Sim/Não
columnsToBool = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
                 "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                 "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]

for col in columnsToBool:
    dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce').astype(bool)

# Aplica One-Hot Encoding nas variáveis categóricas restantes
categoricalColumns = ["gender", "Contract", "PaymentMethod", "InternetService"]
dataframe = pd.get_dummies(dataframe, columns=categoricalColumns, drop_first=False)

# Escalonamento das variáveis numéricas, incluindo a nova feature
numericalColumns = ["tenure", "MonthlyCharges", "TotalCharges", "gastos_medios"]
scaler = StandardScaler()
dataframe[numericalColumns] = scaler.fit_transform(dataframe[numericalColumns])


# --- Análise Exploratória de Dados (EDA) ---
print("--- Gráficos da Análise Exploratória ---")

# Histograma
dataframe["TotalCharges"].hist(bins=30)
plt.title("Distribuição de TotalCharges (Escalonado)")
plt.xlabel("TotalCharges (Escalonado)")
plt.ylabel("Frequência")
plt.show()

# Boxplot
sb.boxplot(x="Churn", y="MonthlyCharges", data=dataframe)
plt.title("Boxplot de MonthlyCharges (Escalonado) por Churn")
plt.show()

# Matriz de Correlação
plt.figure(figsize=(18,12))
# Filtra apenas colunas numéricas para a matriz de correlação
numeric_df = dataframe.select_dtypes(include=[np.number, 'bool'])
sb.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação de Todas as Features")
plt.show()


#------------------------------------------------------------------
# ETAPA 4.2: SELEÇÃO DE FEATURES E REDUÇÃO DE DIMENSIONALIDADE
#------------------------------------------------------------------
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# -- Seleção de Melhores Atributos via SelectKBest --- 
print("\n--- Seleção de Features com SelectKBest ---")

# Separa as features (X) da variável alvo (y)
# 'customerID' é removido pois não é uma feature de treinamento
X = dataframe.drop(columns=['Churn', 'customerID'], errors='ignore')
y = dataframe['Churn']

# Seleciona as 20 melhores features
k_best = 20
selector = SelectKBest(score_func=f_classif, k=k_best)
X_best = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print(f"As {k_best} melhores features selecionadas são:")
print(list(selected_features))


# --- Aplicação de PCA para Visualização 2D --- 
print("\n--- Visualização 2D com PCA ---")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X) # Usa o conjunto X completo

pca_df = pd.DataFrame(data=X_pca, columns=['Componente Principal 1', 'Componente Principal 2'])
pca_df['Churn'] = y.values

plt.figure(figsize=(12, 8))
sb.scatterplot(x='Componente Principal 1', y='Componente Principal 2', hue='Churn', data=pca_df, alpha=0.7)
plt.title('Visualização 2D dos Clientes com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Churn')
plt.grid()
plt.show()