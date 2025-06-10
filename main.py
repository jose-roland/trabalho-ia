import pandas as pd

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
            "No internet service": "Yes",
            "No phone service": "No",
            "0": "No",
            "1": "Yes",
            0: "No",
            1: "Yes"
        }) # Troca valores inconsistentes por Yes ou No

dataframe.loc[dataframe["TotalCharges"].isnull(), "TotalCharges"] = 0
# Preenche as linhas de TotalCharges nulas com 0