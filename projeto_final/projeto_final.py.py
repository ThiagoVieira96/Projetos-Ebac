import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from pycaret.classification import *
from sklearn.metrics import accuracy_score
import cloudpickle

@st.cache_data
def IV(variavel, resposta):
    tab = pd.crosstab(variavel, resposta, margins=True, margins_name='total')

    rótulo_evento = tab.columns[0]
    rótulo_nao_evento = tab.columns[1]

    # Calcular as proporções
    tab['pct_evento'] = tab[rótulo_evento] / tab.loc['total', rótulo_evento]
    tab['pct_nao_evento'] = tab[rótulo_nao_evento] / tab.loc['total', rótulo_nao_evento]

    # Evitar divisão por zero
    tab['woe'] = np.where(tab['pct_nao_evento'] != 0, 
                          np.log(tab['pct_evento'] / tab['pct_nao_evento']),
                          0)

    # Calcular IV parcial
    tab['iv_parcial'] = (tab['pct_evento'] - tab['pct_nao_evento']) * tab['woe']
    return tab['iv_parcial'].sum()

@st.cache_data
def selecionar_numericas(df):
    return list(df.select_dtypes(include=['int64', 'float64']).columns)

@st.cache_data
def remove_outliers(df, columns):
    for col in columns:
        if col != 'mau':  # Correção aqui
            mean = df[col].mean()
            std = df[col].std()
            upper_limit = mean + 2 * std
            df = df[df[col] <= upper_limit]
    return df

@st.cache_data
def renomear_colunas(df):
    # Remove o prefixo e o sublinhado inicial, retornando apenas o nome original da coluna
    return df.rename(columns=lambda x: x.split('__', 1)[-1])

@st.cache_resource
class FiltrarVariaveisIV(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.005):
        self.threshold = threshold
        self.variaveis_selecionadas = []
    
    def fit(self, X, y):
        X_filtrado = X.copy()
        self.variaveis_selecionadas = []

        for col in X_filtrado.columns:
            if X_filtrado[col].nunique() > 6:
                iv = IV(pd.qcut(X_filtrado[col], 5, duplicates='drop'), y)
            else:
                iv = IV(X_filtrado[col], y)

            if iv > self.threshold:
                self.variaveis_selecionadas.append(col)

        return self

    def transform(self, X):
        return X[self.variaveis_selecionadas]


@st.cache_resource
class DummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um DataFrame do pandas.")
        
        colunas_categoricas = X.select_dtypes(include=['object', 'category']).columns
        if len(colunas_categoricas) == 0:
            return X
        
        df_dummies = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True)
        return df_dummies

def plot_histograms(df):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    df['idade'].hist(bins=35, ax=axs[0, 0])
    axs[0, 0].set_title('Histograma de Idade')

    df['tempo_emprego'].hist(bins=35, ax=axs[0, 1])
    axs[0, 1].set_title('Histograma do tempo de emprego')

    df['qt_pessoas_residencia'].hist(bins=35, ax=axs[1, 0])
    axs[1, 0].set_title('Histograma da quantidade de pessoas por residência')

    # Definindo o limite do eixo x para a renda
    renda_mean = df['renda'].mean()
    renda_std = df['renda'].std()
    axs[1, 1].hist(df['renda'], bins=100)
    axs[1, 1].set_xlim(0, renda_mean + (2 * renda_std))
    axs[1, 1].set_title('Histograma da renda')

    # Renderizando os gráficos com o Streamlit
    st.pyplot(fig)




def main():
    st.set_page_config(page_title='Projeto Final',
                       page_icon= 'https://seeklogo.com/images/E/ebac-logo-CC73A39D07-seeklogo.com.png',
                       layout="wide",
                       initial_sidebar_state='expanded')
    
    st.write('''# Projeto Final 💻
             
Este é o projeto final do curso de ciência de dados da EBAC. Neste projeto será exibido o pipeline criado no exercício anterior
e também as transformações feitas com pycaret, além de disponibilizar ambos modelos finais para download.

Informações adicionais estarão linkadas no repositório do Github, juntamente com o código finte desta página.

Este projeto, assim como o arquivo python notebook deste capítulo, consiste em uma análise de crédito, onde serão feitas diversas transformações na base de dados,
sendo estas: substituição de valores NaN, remoção de outliers, filtro de variáveis com base no Information Value, criação de dummies para variáveis categóricas e 
criação de um PCA com dimensionalidade 5. Estas transformações serão automatizadas via pipeline.

Por fim, essa pipeline será disponibilizada para download.
             ''')
    
    data_url = 'https://raw.githubusercontent.com/ThiagoVieira96/Projetos-Ebac/main/projeto_final/credit_comprimido.ftr'

    
    df = pd.read_feather(data_url)
    
    # Conteúdo da barra lateral
    st.sidebar.image ('https://seeklogo.com/images/E/ebac-logo-CC73A39D07-seeklogo.com.png')
    st.sidebar.header("Projeto Final")
    st.sidebar.write("""
    Esta aplicação cria uma pipeline, aplica ela ao conjunto de dados em questão e a disponibiliza para download.
    O arquivo de dados utilizado pode ser encontrado [aqui](https://raw.githubusercontent.com/ThiagoVieira96/Projetos-Ebac/main/projeto_final/credit_comprimido.ftr).
    """)
    
    df_oot = df[df['data_ref'] > (df['data_ref'].max() - pd.DateOffset(months= 3))]
    df_test = df[~(df['data_ref'] > (df['data_ref'].max() - pd.DateOffset(months= 3)))]
    
    st.write(df.head())
    
    tab1, tab2 = st.tabs(["📋 Análises" , "💻 Pipeline" ])
    
    with tab1:
        st.write('# Descritiva Univariada ')
        st.write('''Aqui faremos as primeiras análies univariadas e bivariadas do dataframe, antes de se criar o pipeline''')
        st.write('Primeiramente faremos uma análise univaridada das variáveis numéricas. Seguem abaixo os histogramas:')
        
        
        plot_histograms(df_test)
        
        st.write('''Agora iremos fazer análises univariadas das variáveis categóricas, segue abaixo os gráficos:''')
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Gráfico 1 - Distribuição por sexo
        df_test['sexo'].value_counts().plot(kind='bar', ax=axs[0, 0])
        axs[0, 0].set_title('Distribuição por sexo')
        axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)

        # Gráfico 2 - Distribuição por posse de imóvel
        df_test['posse_de_imovel'].value_counts().plot(kind='bar', ax=axs[0, 1])
        axs[0, 1].set_title('Distribuição por posse de imóvel')
        axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45)

        # Gráfico 3 - Distribuição por tipo de renda
        df_test['tipo_renda'].value_counts().plot(kind='bar', ax=axs[1, 0])
        axs[1, 0].set_title('Distribuição por tipo de renda')
        axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=45)

        # Gráfico 4 - Distribuição por grau de educação
        df_test['educacao'].value_counts().plot(kind='bar', ax=axs[1, 1])
        axs[1, 1].set_title('Distribuição por grau de educação')
        axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45)

        # Gráfico 5 - Distribuição por tipo de residência
        df_test['tipo_residencia'].value_counts().plot(kind='bar', ax=axs[2, 0])
        axs[2, 0].set_title('Distribuição por tipo de residência')
        axs[2, 0].set_xticklabels(axs[2, 0].get_xticklabels(), rotation=45)

        # Gráfico 6 - Distribuição por estado civil
        df_test['estado_civil'].value_counts().plot(kind='bar', ax=axs[2, 1])
        axs[2, 1].set_title('Distribuição por estado civil')
        axs[2, 1].set_xticklabels(axs[2, 1].get_xticklabels(), rotation=45)

        # Ajustando o layout
        plt.subplots_adjust(hspace=0.8)
        plt.tight_layout()

        # Exibindo os gráficos no Streamlit
        st.pyplot(fig)
        
        st.write('# Descritiva Bivariada ')
        st.write('Agora iremos fazer a análise bivariada, como "mau" é a variável resposta, iremos fazer todos os gráficos em função dela.')
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        sns.countplot(x='sexo', hue='mau', data=df_test, ax=axs[0, 0])
        axs[0, 0].set_title('Sexo vs Mau')

        sns.countplot(x='estado_civil', hue='mau', data=df_test, ax=axs[0, 1])
        axs[0, 1].set_title('Estado Civil vs Mau')
        axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=45)

        sns.countplot(x='posse_de_imovel', hue='mau', data=df_test, ax=axs[1, 0])
        axs[1, 0].set_title('Posse de Imóvel vs Mau')

        sns.countplot(x='tipo_renda', hue='mau', data=df_test, ax=axs[1, 1])
        axs[1, 1].set_title('Tipo de Renda vs Mau')
        axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45)


        plt.subplots_adjust(hspace=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        sns.boxplot(x='mau', y='idade', data=df_test, ax=axs[0, 0])
        axs[0, 0].set_title('Idade vs Mau')

        sns.boxplot(x='mau', y='renda', data=df_test, ax=axs[0, 1])
        axs[0, 1].set_title('Renda vs Mau')

        sns.boxplot(x='mau', y='tempo_emprego', data=df_test, ax=axs[1, 0])
        axs[1, 0].set_title('Tempo de Emprego vs Mau')

        sns.boxplot(x='mau', y='tempo_emprego', data=df_test, ax=axs[1, 1])
        axs[1, 1].set_title('Qt Pessoas Residência vs Mau')

        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
        
        
    
    with tab2:
        st.write('# Pipeline')
        st.write('Agora que já fizemos todas as análies peritnentes, iremos fazer o pipeline para automatizar as transformações na base de dados.') 
        st.write('''Como mencinado anteriormente, este pipeline consiste em substituir valores nulos, remover outliers, filtrar as variáveis com base no seu IV,
    criar variáveis dummy para as variáveis categóricas e aplicar um PCA de dimensionalidade 5.
    
Também iremos criar um modelo de random forest para treinar usando o output deste pipeline, sendo que iremos medir a qualidade deste
modelo por meio de sua acurácia.''')

        df_test_2 = df_test.drop(columns=['index', 'data_ref'])
        colunas_numericas = selecionar_numericas(df_test_2)
        
        # Get X and y
        X = df_test_2.drop(columns=['mau'])
        y = df_test_2['mau']
        
        # Combine X and y into a single DataFrame
        data = X.copy()
        data['mau'] = y
        
        # Apply remove_outliers to the combined data
        data = remove_outliers(data, columns=colunas_numericas)
        
        # Now separate X and y after outlier removal
        X = data.drop(columns=['mau'])
        y = data['mau']

        preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), colunas_numericas)
        ], 
        remainder='passthrough')
        preprocessor.set_output(transform= 'pandas')

        iv_selector = FiltrarVariaveisIV()
        pca_pipe = PCA(n_components=5)
        dummy_transformer = DummyTransformer()        
        

        pipe = Pipeline(steps=[
            ('inputer', preprocessor),
            ('renomear', FunctionTransformer(renomear_colunas, validate=False)),
            # Remove the 'outliers' step from the pipeline
            #('outliers', outlier_removal),
            ('filter', iv_selector),
            ('dummies', dummy_transformer),
            ('pca', pca_pipe)
        ])
        
        #st.write(pipe.fit(X, y))
        
        X = df_test_2.drop(columns=['mau'])

        y = df_test_2['mau']

        # Aplicar o pipeline somente em X (sem a variável 'mau')

        X_transformed = pipe.fit_transform(X, y)

        # Agora podemos treinar o modelo com X_transformed e y

        model = RandomForestClassifier(random_state=42)

        model.fit(X_transformed, y)
        
        # Exibir o modelo treinado no Streamlit

        st.write("Modelo RandomForestClassifier treinado:")

        st.write(model)   
        
        y_model = model.predict(X_transformed)
    
        acuracia = accuracy_score(y, y_model)
        
        st.write(f"""Agora com o modelo treinado, temos uma acurácia de: {acuracia}""")
        
        st.write('Por fim, a pipeline criada está disponível para download no botão abaixo.')
        
        # Salvar o pipeline usando cloudpickle
        with open("pipeline.pkl", "wb") as f:
            cloudpickle.dump(pipe, f)

        # Disponibilizar o arquivo para download
        with open("pipeline.pkl", "rb") as file:
            st.download_button(
                label="Download Pipeline",
                data=file,
                file_name="pipeline.pkl",
                mime="application/octet-stream"
            )
                   
        
        

        
if __name__ == '__main__':
    main()