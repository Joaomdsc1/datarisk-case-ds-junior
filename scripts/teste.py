import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import warnings


# Definindo os caminhos para os arquivos de dados
PATH_CADASTRAL = 'data/base_cadastral.csv'
PATH_INFO = 'data/base_info.csv'
PATH_DEV = 'data/base_pagamentos_desenvolvimento.csv'
PATH_TEST = 'data/base_pagamentos_teste.csv'

# Colunas que são datas
date_cols_dev = ['DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO']
date_cols_test = ['DATA_EMISSAO_DOCUMENTO', 'DATA_VENCIMENTO']
date_cols_cadastral = ['DATA_CADASTRO']

# Carregando as bases com separador ';' e tratando datas
df_cadastral = pd.read_csv(PATH_CADASTRAL, sep=';', parse_dates=date_cols_cadastral)
df_info = pd.read_csv(PATH_INFO, sep=';', parse_dates=['SAFRA_REF'])
df_dev = pd.read_csv(PATH_DEV, sep=';', parse_dates=date_cols_dev)
df_test = pd.read_csv(PATH_TEST, sep=';', parse_dates=date_cols_test)

# A coluna SAFRA_REF nas bases de pagamento também é uma data
df_dev['SAFRA_REF'] = pd.to_datetime(df_dev['SAFRA_REF'])
df_test['SAFRA_REF'] = pd.to_datetime(df_test['SAFRA_REF'])

print("\nBases de dados carregadas:")
print(f"Base Cadastral: {df_cadastral.shape}")
print(f"Base Info: {df_info.shape}")
print(f"Base Desenvolvimento: {df_dev.shape}")
print(f"Base Teste: {df_test.shape}")


# # # CRIAÇÃO DA VARIÁVEL ALVO (INADIMPLÊNCIA)

# A inadimplência é definida por um atraso de 5 dias ou mais
df_dev['DIAS_ATRASO'] = (df_dev['DATA_PAGAMENTO'] - df_dev['DATA_VENCIMENTO']).dt.days
df_dev['INADIMPLENTE'] = np.where(df_dev['DIAS_ATRASO'] >= 5, 1, 0)

# A coluna DATA_PAGAMENTO não existe no teste e não deve ser usada como feature
df_dev = df_dev.drop(columns=['DATA_PAGAMENTO', 'DIAS_ATRASO'])

print("\nVariável alvo 'INADIMPLENTE' criada.")
print("Distribuição da variável alvo:")
print(df_dev['INADIMPLENTE'].value_counts(normalize=True))


# # # FUNÇÃO PARA UNIFICAR E CRIAR VARIÁVEIS (FEATURE ENGINEERING)

def feature_engineering(df_pagamentos, df_cadastral, df_info):
    df = pd.merge(df_pagamentos, df_cadastral, on='ID_CLIENTE', how='left')
    df = pd.merge(df, df_info, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')

    df['TEMPO_DE_CLIENTE_DIAS'] = (df['SAFRA_REF'] - df['DATA_CADASTRO']).dt.days
    df['PRAZO_PAGAMENTO_DIAS'] = (df['DATA_VENCIMENTO'] - df['DATA_EMISSAO_DOCUMENTO']).dt.days
    df['DIA_SEMANA_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.dayofweek
    df['MES_VENCIMENTO'] = df['DATA_VENCIMENTO'].dt.month
    df['E_PESSOA_FISICA'] = np.where(df['FLAG_PF'] == 'X', 1, 0)

    free_emails = ['gmail', 'hotmail', 'yahoo', 'outlook', 'live', 'msn', 'uol', 'bol', 'terra']
    # Tratamento do tipo de e-mail
    df['DOMINIO_EMAIL_TIPO'] = df['DOMINIO_EMAIL'].apply(
        lambda x: 'gratuito' if str(x).split('.')[0] in free_emails else 'corporativo'
    )
    
    df['VALOR_PELA_RENDA'] = df['VALOR_A_PAGAR'] / (df['RENDA_MES_ANTERIOR'] + 1)
    df['RENDA_POR_FUNCIONARIO'] = df['RENDA_MES_ANTERIOR'] / (df['NO_FUNCIONARIOS'] + 1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

df_train_full = feature_engineering(df_dev, df_cadastral, df_info)
df_test_full = feature_engineering(df_test, df_cadastral, df_info)

print("\nEngenharia de variáveis concluída.")


# # # PRÉ-PROCESSAMENTO E DEFINIÇÃO DO PIPELINE

TARGET = 'INADIMPLENTE'
DROP_FEATURES = ['ID_CLIENTE', 'SAFRA_REF', 'DATA_CADASTRO', 'DATA_EMISSAO_DOCUMENTO', 'DATA_VENCIMENTO', 'FLAG_PF', 'DOMINIO_EMAIL']

features = [col for col in df_train_full.columns if col not in [TARGET] + DROP_FEATURES]
categorical_features = df_train_full[features].select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = df_train_full[features].select_dtypes(include=np.number).columns.tolist()

for col in ['DDD', 'CEP_2_DIG', 'DIA_SEMANA_VENCIMENTO', 'MES_VENCIMENTO']:
    if col in numerical_features:
        numerical_features.remove(col)
        categorical_features.append(col)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

print("\nPipeline de pré-processamento definido.")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


# # # VALIDAÇÃO CRUZADA, TREINAMENTO E AVALIAÇÃO DO MODELO

lgbm = lgb.LGBMClassifier(random_state=42)
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm)])

# Ordenar por SAFRA_REF é crucial para o TimeSeriesSplit funcionar corretamente
df_train_full = df_train_full.sort_values(by='SAFRA_REF').reset_index(drop=True)
X = df_train_full[features]
y = df_train_full[TARGET]

print("\nIniciando a validação cruzada com TimeSeriesSplit...")
tscv = TimeSeriesSplit(n_splits=5)
auc_scores = []

for fold, (train_index, val_index) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model_pipeline.fit(X_train, y_train)
    y_pred_proba = model_pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print("-" * 30)
print(f"Média AUC da Validação Cruzada: {np.mean(auc_scores):.4f}")
print(f"Desvio Padrão AUC: {np.std(auc_scores):.4f}")
print("-" * 30)

print("\nTreinando o modelo final com todos os dados de desenvolvimento...")
model_pipeline.fit(X, y)
print("Treinamento final concluído.")


# # # ANÁLISE DE IMPORTÂNCIA DAS VARIÁVEIS

ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)
importances = model_pipeline.named_steps['classifier'].feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nAs 20 variáveis mais importantes para o modelo:")
print(feature_importance_df.head(20))

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
plt.title('As 20 Variáveis Mais Importantes para o Modelo')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nGráfico de importância das variáveis salvo como 'feature_importance.png'")


# # # GERAÇÃO DO ARQUIVO DE SUBMISSÃO

print("\nGerando previsões para a base de teste...")
X_test = df_test_full[features]
test_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
print("Previsões geradas.")

submission_df = pd.DataFrame({
    'ID_CLIENTE': df_test_full['ID_CLIENTE'],
    'SAFRA_REF': df_test_full['SAFRA_REF'].dt.strftime('%Y-%m-%d'),
    'PROBABILIDADE_INADIMPLENCIA': test_probabilities 
})

submission_filename = 'submissao_case.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nArquivo de submissão '{submission_filename}' criado com sucesso!")
print(submission_df.head())