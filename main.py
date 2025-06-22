import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb

columnsToNumeric = ["SeniorCitizen",
                    "Partner",
                    "Dependents",
                    "PhoneService",
                    "MultipleLines",
                    "OnlineSecurity",
                    "OnlineBackup",
                    "DeviceProtection",
                    "TechSupport",
                    "StreamingTV",
                    "StreamingMovies",
                    "PaperlessBilling",
                    "TotalCharges",
                    "Churn"] # Lista das colunas que terão seus tipos convertidos de object para numeric

categoricalColumns = ["gender", "Contract", "PaymentMethod"]
# Lista das colunas que vão passar pelo one-hot encoding

numericalColumns = ["tenure", "MonthlyCharges", "TotalCharges"]

pd.set_option('future.no_silent_downcasting', True)
# Evita warning de que função replace mude dados de forma automática (downcasting)

dataframe = pd.read_csv("customerchurn.csv")
# Definindo o Dataframe

scaler = StandardScaler()
# Definindo o StandardScaler para escalonar variáveis

dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
# Converte os dados de TotalCharges para numéricoso

for column in dataframe.columns:
    if dataframe[column].dtype == object:
        dataframe[column] = dataframe[column].str.strip()
        # Remove espaços em branco dos valores, cajo haja algum

        dataframe[column] = dataframe[column].replace({
            "No phone service": 0,
            "No": 0,
            "Yes": 1,
            "0": 0,
            "1": 1
        }) # Troca valores por 0 ou 1

dataframe.loc[dataframe["TotalCharges"].isnull(), "TotalCharges"] = 0
# Preenche as linhas de TotalCharges nulas com 0

for i in columnsToNumeric:
    dataframe[i] = pd.to_numeric(dataframe[i], errors="coerce")
    # Conversão de tipo para numeric

    if i != "TotalCharges":
        dataframe[i] = dataframe[i].astype(bool)
        # "Castando" as colunas para tipo bool (exceto TotalCharges)

dataframe = pd.get_dummies(dataframe, columns=categoricalColumns, drop_first=False)
# Aplica one-hot encoding para todas as colunas no categoricalColumns

internet_dummies = pd.get_dummies(dataframe["InternetService"], prefix="InternetService", drop_first=True)
# Aplica one-hot encoding apenas para a coluna InternetService, descartando a primeira coluna (No)

dataframe = pd.concat([dataframe, internet_dummies], axis=1)
# Adiciona as colunas dummies (one-hot encoding) no dataframe

dataframe.drop(columns=["InternetService"], inplace=True)
# Remove a coluna original do dataframe, deixando apenas as que passaram pelo one-hot encoding

dataframe[numericalColumns] = scaler.fit_transform(dataframe[numericalColumns])
# Escalonamento de variáveis

dataframe["TotalCharges"].hist(bins=30)
plt.title("Distribuição de TotalCharges")
plt.xlabel("TotalCharges")
plt.ylabel("Frequência")
plt.show()

sb.boxplot(x="Churn", y="MonthlyCharges", data=dataframe)
plt.title("Boxplot de MonthlyCharges por Churn")
plt.show()

plt.figure(figsize=(12,8))
sb.heatmap(dataframe.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlação")
plt.show()