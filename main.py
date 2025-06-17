import pandas as pd

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

pd.set_option('future.no_silent_downcasting', True)
# Evita warning de que função replace mude dados de forma automática (downcasting)

dataframe = pd.read_csv("customerchurn.csv")
# Leitura do CSV

dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
# Converte os dados de TotalCharges para numéricos

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