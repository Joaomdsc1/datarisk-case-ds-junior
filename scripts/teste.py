import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import warnings

# # # CARREGAMENTO E PREPARAÇÃO INICIAL DOS DADOS

# Definindo os caminhos para os arquivos de dados
PATH_CADASTRAL = 'data/base_cadastral.csv'
PATH_INFO = 'data/base_info.csv'
PATH_DEV = 'data/base_pagamentos_desenvolvimento.csv'
PATH_TEST = 'data/base_pagamentos_teste.csv'

# Colunas que são datas
date_cols_dev = ['DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO']
date_cols_test = ['DATA_EMISSAO_DOCUMENTO', 'DATA_VENCIMENTO']
date_cols_cadastral = ['DATA_CADASTRO']

# Carregando as bases
df_cadastral = pd.read_csv(PATH_CADASTRAL, sep=';', parse_dates=date_cols_cadastral)
df_info = pd.read_csv(PATH_INFO, sep=';', parse_dates=['SAFRA_REF'])
df_dev = pd.read_csv(PATH_DEV, sep=';', parse_dates=date_cols_dev)
df_test = pd.read_csv(PATH_TEST, sep=';', parse_dates=date_cols_test)

# A coluna SAFRA_REF nas bases de pagamento também é uma data
df_dev['SAFRA_REF'] = pd.to_datetime(df_dev['SAFRA_REF'])
df_test['SAFRA_REF'] = pd.to_datetime(df_test['SAFRA_REF'])

print("Bases de dados carregadas:")
print(f"Base Cadastral: {df_cadastral.shape}")
print(f"Base Info: {df_info.shape}")
print(f"Base Desenvolvimento: {df_dev.shape}")
print(f"Base Teste: {df_test.shape}")


# # # CRIAÇÃO DA VARIÁVEL ALVO (INADIMPLÊNCIA)

# A inadimplência é definida por um atraso de 5 dias ou mais
df_dev['DIAS_ATRASO'] = (df_dev['DATA_PAGAMENTO'] - df_dev['DATA_VENCIMENTO']).dt.days
df_dev['INADIMPLENTE'] = np.where(df_dev['DIAS_ATRASO'] >= 5, 1, 0)

# A coluna DATA_PAGAMENTO não existe na base de teste e não deve ser usada como feature
# A coluna DIAS_ATRASO é uma consequência da variável alvo, portanto também é removida
df_dev = df_dev.drop(columns=['DATA_PAGAMENTO', 'DIAS_ATRASO'])

print("\nVariável alvo 'INADIMPLENTE' criada.")
print("Distribuição da variável alvo:")
print(df_dev['INADIMPLENTE'].value_counts(normalize=True))

# # # UNIFICAR E CRIAR VARIÁVEIS

def feature_engineering(df_pagamentos, df_cadastral, df_info):

    # Unificando as bases
    df = pd.merge(df_pagamentos, df_cadastral, on='ID_CLIENTE', how='left')
    df = pd.merge(df, df_info, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')

    # --- Engenharia de Variáveis ---

    # Variáveis de tempo
    df['TEMPO_DE_CLIENTE_DIAS'] = (df['SAFRA_REF'] - df['DATA_CADASTRO']).dt.days
    df['PRAZO_PAGAMENTO_DIAS'] = (df['DATA_VENCIMENTO'] - df['DATA_EMISSAO_DOCUMENTO']).dt.days
    df['DIA_SEMANA_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.dayofweek
    df['MES_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.month

    # Tratamento da flag de Pessoa Física
    df['E_PESSOA_FISICA'] = np.where(df['FLAG_PF'] == 'X', 1, 0)

    # Tratamento do domínio de e-mail
    free_emails = ['gmail', 'hotmail', 'yahoo', 'outlook', 'live', 'msn', 'uol', 'bol', 'terra']
    df['DOMINIO_EMAIL_TIPO'] = df['DOMINIO_EMAIL'].apply(
        lambda x: 'gratuito' if str(x).split('.')[0] in free_emails else 'corporativo'
    )
    
    # Variáveis de proporção
    # Adicionar 1 para evitar divisão por zero
    df['VALOR_PELA_RENDA'] = df['VALOR_A_PAGAR'] / (df['RENDA_MES_ANTERIOR'] + 1)
    df['RENDA_POR_FUNCIONARIO'] = df['RENDA_MES_ANTERIOR'] / (df['NO_FUNCIONARIOS'] + 1)

    # Tratando valores infinitos que podem surgir da divisão
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

# Aplicando a função nas bases de desenvolvimento e teste
df_train_full = feature_engineering(df_dev, df_cadastral, df_info)
df_test_full = feature_engineering(df_test, df_cadastral, df_info)

print("\nEngenharia de variáveis concluída.")
print(f"Formato final da base de treino: {df_train_full.shape}")
print(f"Formato final da base de teste: {df_test_full.shape}")

# # # 5. PRÉ-PROCESSAMENTO E DEFINIÇÃO DO PIPELINE DE MODELAGEM

# Definindo a variável alvo e as features a serem usadas
TARGET = 'INADIMPLENTE'
# Features a serem removidas (identificadores, datas, etc.)
DROP_FEATURES = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_CADASTRO', 'DATA_EMISSAO_DOCUMENTO', 'DATA_VENCIMENTO', 'FLAG_PF', 'DOMINIO_EMAIL']

# Identificando tipos de colunas para o pré-processamento
features = [col for col in df_train_full.columns if col not in [TARGET] + DROP_FEATURES]
categorical_features = df_train_full[features].select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = df_train_full[features].select_dtypes(include=np.number).columns.tolist()

# Convertendo colunas que deveriam ser categóricas
for col in ['DDD', 'CEP_2_DIG', 'DIA_SEMANA_VENCIMENTO', 'MES_VENCIMENTO']:
    if col in numerical_features:
        numerical_features.remove(col)
        categorical_features.append(col)
        
print(f"\nFeatures numéricas ({len(numerical_features)}): {numerical_features}")
print(f"Features categóricas ({len(categorical_features)}): {categorical_features}")

# Criando o pipeline de pré-processamento
# Para features numéricas, preenchemos nulos com a mediana e escalonamos
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Para features categóricas, preenchemos nulos com 'missing' e aplicamos OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinando os transformers em um único ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Descarta colunas não especificadas
)

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")

# # # TREINAMENTO E AVALIAÇÃO DO MODELO

# Definindo o modelo
lgbm = lgb.LGBMClassifier(random_state=42)

# Criando o pipeline final que inclui o pré-processamento e o classificador
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm)
])

# Separando features (X) e alvo (y)
X = df_train_full[features]
y = df_train_full[TARGET]

# Divisão dos dados em treino e validação (80/20 split)
# Usar stratify=y é bom para garantir a proporção da variável alvo em ambos os sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTamanho do conjunto de treino: {X_train.shape}")
print(f"Tamanho do conjunto de validação: {X_val.shape}")

# Treinando o pipeline
print("\nIniciando o treinamento do modelo...")
model_pipeline.fit(X_train, y_train)
print("Treinamento concluído.")

# Avaliando o modelo no conjunto de validação
y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)

print(f"\nAUC no conjunto de validação: {auc:.4f}")

# # # GERAR PREVISÕES E CRIAR ARQUIVO

print("\nGerando previsões para a base de teste...")
# Selecionando as mesmas features na base de teste
X_test = df_test_full[features]

# Usando o pipeline treinado para prever as probabilidades na base de teste
test_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
print("Previsões geradas.")

# Criando o arquivo de submissão no formato exigido
submission_df = pd.DataFrame({
    'ID_CLIENTE': df_test_full['ID_CLIENTE'],
    'SAFRA_REF': df_test_full['SAFRA_REF'].dt.strftime('%Y-%m-%d'),
    'PROBABILIDADE_INADIMPLENCIA': test_probabilities
})

# Salvando o arquivo CSV
submission_filename = 'submissao_case.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nArquivo de submissão '{submission_filename}' criado com sucesso!")
print(submission_df.head())
