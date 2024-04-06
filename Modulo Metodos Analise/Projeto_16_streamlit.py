import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import streamlit as st
from streamlit.components.v1 import components
import os
from ydata_profiling import ProfileReport

output_folder_path = "./output"
st.set_page_config(page_title = 'Projeto módulo 16',
                    page_icon='https://cdn.icon-icons.com/icons2/1446/PNG/512/22261pandaface_98765.png',
                    layout='wide')

st.write('# Projeto Pandas Avançado')
st.write('Este é o projeto do módulo 16 do curso de ciência de dados da EBAC.')
st.markdown('------')
st.write('''Este notebook conterá o projeto final do módulo 16 do curso de ciência de dados da EBAC,
         nele, iremos analisar a base de dados de renda que já foi utilizada em módulos passados,
         aqui iremos fazer uma análise nos moldes do método CRISP - DM,
         onde todas as etapas necessárias,
         desde entendimento dos dados até implementação será feita e discutida no mesmo notebook.''')
st.write('Neste arquivo irão conter os gráficos referentes a base de dados, os modelos de regressão e as árvores de decisão, lembrando que a variável resposta é sempre a renda.')
st.write('''A base de dados em questão contem diversas colunas com informações sobre os clientes, segue abaixo uma explicação sobre estes:
''')
st.markdown('''| Variável                | Descrição                                           | Tipo         |
| ----------------------- |:---------------------------------------------------:| ------------:|
| data_ref                | Data de referência de coleta das variáveis          | Object       |
| id_cliente              | Código de identificação do cliente                  | Int 64       |
| sexo                    | Sexo do cliente                                     | Object       |
| posse_de_veiculo        | Indica se o cliente possui veículo                  | Bool         |
| posse_de_imovel         | Indica se o cliente possui imóvel                   | Bool         |
| qtd_filhos              | Quantidade de filhos do cliente                     | Int 64       |
| tipo_renda              | Tipo de renda do cliente                            | Object       |
| educacao                | Grau de instrução do cliente                        | Object       |
| estado_civil            | Estado civil do cliente                             | Object       |
| tipo_residencia         | Tipo de residência do cliente (própria, alugada etc)| Object       |
| idade                   | Idade do cliente                                    | Int 64       |
| tempo_emprego           | Tempo no emprego atual                              | Float 64     |
| qt_pessoas_residencia   | Quantidade de pessoas que moram na residência       | Float 64     |
| renda                   | Renda em reais                                      | Float 64     |''')
st.write('''Antes de ser feita qualquer análise,
         todas as colunas foram convertidas para valores numéricos,
         criando-se variáveis dummy quando necessário, além disso,
         as duas primeiras colunas "data_ref" e "id_cliente" foram excluídas,
         já que só iriam atrapalhar as análises.''')
st.markdown('------')

renda = pd.read_csv('https://raw.githubusercontent.com/ThiagoVieira96/Projetos-Ebac/main/Modulo%20Metodos%20Analise/previsao_de_renda.csv')
renda_encoded = (pd.get_dummies(renda, columns=['sexo', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia'], drop_first=True)
                 .drop(['Unnamed: 0', 'data_ref', 'id_cliente'], axis=1)
                 .reset_index(drop=True))
renda_encoded[['posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos', 'idade',
         'sexo_M',
       'tipo_renda_Bolsista', 'tipo_renda_Empresário',
       'tipo_renda_Pensionista', 'tipo_renda_Servidor público',
       'educacao_Pós graduação', 'educacao_Secundário',
       'educacao_Superior completo', 'educacao_Superior incompleto',
       'estado_civil_Separado', 'estado_civil_Solteiro', 'estado_civil_União',
       'estado_civil_Viúvo', 'tipo_residencia_Casa',
       'tipo_residencia_Com os pais', 'tipo_residencia_Comunitário',
       'tipo_residencia_Estúdio', 'tipo_residencia_Governamental']] = renda_encoded[['posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos', 'idade',
         'sexo_M',
       'tipo_renda_Bolsista', 'tipo_renda_Empresário',
       'tipo_renda_Pensionista', 'tipo_renda_Servidor público',
       'educacao_Pós graduação', 'educacao_Secundário',
       'educacao_Superior completo', 'educacao_Superior incompleto',
       'estado_civil_Separado', 'estado_civil_Solteiro', 'estado_civil_União',
       'estado_civil_Viúvo', 'tipo_residencia_Casa',
       'tipo_residencia_Com os pais', 'tipo_residencia_Comunitário',
       'tipo_residencia_Estúdio', 'tipo_residencia_Governamental']].astype(int)
       

# Barra lateral para escolher gráficos
st.sidebar.write('# Menu')
selected_charts = st.sidebar.multiselect('Selecione os gráficos a serem exibidos', ['Gráficos ao longo do tempo', 'Gráficos bivariados'])


# Gráficos ao longo do tempo
if 'Gráficos ao longo do tempo' in selected_charts:
    st.write('## Gráficos ao longo do tempo')
    fig, ax = plt.subplots(8, 1, figsize=(10, 70))
    renda[['posse_de_imovel', 'renda']].plot(kind='hist', ax=ax[0])
    sns.lineplot(x='data_ref', y='renda', hue='posse_de_imovel', data=renda, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo', data=renda, ax=ax[2])
    ax[2].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='qtd_filhos', data=renda, ax=ax[3])
    ax[3].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='tipo_renda', data=renda, ax=ax[4])
    ax[4].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='educacao', data=renda, ax=ax[5])
    ax[5].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='estado_civil', data=renda, ax=ax[6])
    ax[6].tick_params(axis='x', rotation=45)
    sns.lineplot(x='data_ref', y='renda', hue='tipo_residencia', data=renda, ax=ax[7])
    ax[7].tick_params(axis='x', rotation=45)
    sns.despine()
    st.pyplot(plt)
    st.markdown('------')

# Gráficos bivariados
if 'Gráficos bivariados' in selected_charts:
    st.write('## Gráficos bivariados')
    fig, ax = plt.subplots(7, 1, figsize=(10, 50))
    sns.barplot(x='posse_de_imovel', y='renda', data=renda, ax=ax[0])
    sns.barplot(x='posse_de_veiculo', y='renda', data=renda, ax=ax[1])
    sns.barplot(x='qtd_filhos', y='renda', data=renda, ax=ax[2])
    sns.barplot(x='tipo_renda', y='renda', data=renda, ax=ax[3])
    sns.barplot(x='educacao', y='renda', data=renda, ax=ax[4])
    sns.barplot(x='estado_civil', y='renda', data=renda, ax=ax[5])
    sns.barplot(x='tipo_residencia', y='renda', data=renda, ax=ax[6])
    sns.despine()
    st.pyplot(plt)
    st.markdown('------')
    
selected_analysis = st.sidebar.multiselect('Selecione as análises a serem exibidas',
                                         ['Regressão da base de dados completa',
                                          'Regressão da base de dados tratada',
                                          'Árvore da base de dados completa',
                                          'Árvore da base de dados tratada'])

if 'Regressão da base de dados completa' in selected_analysis:
    X = renda_encoded.drop(columns=['renda']).fillna(method='bfill').copy()
    y = renda_encoded['renda']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = sm.OLS(y_train, X_train).fit()
    
  # Exibindo resumo da regressão
    st.write('## Regressão da base de dados completa')
    st.text(model.summary())

    # Plotando gráficos de diagnóstico
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    sm.graphics.plot_regress_exog(model, 'idade', fig=fig)
    plt.subplots_adjust(wspace=0.5)
    for axis in ax.flatten():
      axis.tick_params(axis='x', labelrotation=30)
    st.pyplot(plt)
    st.markdown('------')

if 'Regressão da base de dados tratada' in selected_analysis:
  X_2 = renda_encoded.drop(columns=['posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos',
       'tipo_renda_Bolsista', 'tipo_renda_Servidor público',
       'educacao_Pós graduação', 'educacao_Secundário',
       'educacao_Superior completo', 'educacao_Superior incompleto',
       'estado_civil_Viúvo', 'tipo_residencia_Casa',
       'tipo_residencia_Com os pais', 'tipo_residencia_Comunitário',
       'tipo_residencia_Estúdio', 'tipo_residencia_Governamental', 'renda', 'estado_civil_Separado', 'estado_civil_Solteiro',
       'estado_civil_União', 'qt_pessoas_residencia']).fillna(method='bfill').copy()
  y_2 = np.log(renda_encoded['renda'])

  X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.3, random_state=42)

  model_2 = sm.OLS(y_2_train, X_2_train).fit()
  st.write('## Regressão da base de dados tratada')
  st.write('''Obs: Essa foi a regressão com o maior valor de r-quadrado obtido, para isso,
           além de se descartar as colunas com p-value maiores que 5%,
           foi aplicado o logaritmo na coluna de renda.(para mais detalhes conferir o notebook.)''')
  st.text(model_2.summary())
  
  fig, ax = plt.subplots(2, 2, figsize=(12, 10))
  sm.graphics.plot_regress_exog(model_2, 'idade', fig=fig)
  plt.subplots_adjust(wspace=0.5)
  for axis in ax.flatten():
    axis.tick_params(axis='x', labelrotation=30)
  st.pyplot(plt)
  st.markdown('------')

if 'Árvore da base de dados completa' in selected_analysis:
  st.write('''Essa é a arvore de decisão feita se utlizando a base de dados completa, sem nenhum tipo de tratamento,
           nela, a profundidade máxima (max_depth) foi ajustada de modo a se obter o maior valor de
           r-quadrado entre os valores previstos e os valores reais.''')
  X = renda_encoded.drop(columns=['renda']).fillna(method='bfill').copy()
  y = renda_encoded['renda']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
  arvore = DecisionTreeRegressor(max_depth=7 ,random_state= 42)
  arvore.fit(X_train, y_train)
  previsao = arvore.predict(X_test)
  r_square = r2_score(y_test, previsao)
  
  st.markdown(f'O valor de r-quadrado obtido dela foi {r_square:.3f}, com uma profundidade de {arvore.get_depth()}')
  plt.figure(figsize=(25, 10))
  plot_tree(arvore,
          filled=True,
          feature_names=X.columns)
  st.write('## Árvore de Decisão da base de dados completa')
  st.pyplot(plt)
  st.markdown('------')

  if st.button("Salvar Imagem da Árvore de Decisão"):
        os.makedirs(output_folder_path, exist_ok=True)
        
        file_path = os.path.join(output_folder_path, "arvore_decisao_completa.png")
        plt.savefig(file_path)
        st.success(f"Imagem salva em: {file_path}")
  
if 'Árvore da base de dados tratada' in selected_analysis:
  st.write('''Essa é a árvore de decisão tratada, onde foram retiradas todas as colunas cujos p-values são maiores que 5%,
           estes valores podem ser vistos na regressão da base de dados completa.
           Assim como na outra árvore de decisão, a profundidade máxima foi feita de modo a gerar o maior valor
           de r-quadrado entre os valores de renda previstos e os reais.''')
  
  X_2 = renda_encoded.drop(columns=['posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos',
       'tipo_renda_Bolsista', 'tipo_renda_Servidor público',
       'educacao_Pós graduação', 'educacao_Secundário',
       'educacao_Superior completo', 'educacao_Superior incompleto',
       'estado_civil_Viúvo', 'tipo_residencia_Casa',
       'tipo_residencia_Com os pais', 'tipo_residencia_Comunitário',
       'tipo_residencia_Estúdio', 'tipo_residencia_Governamental', 'renda', 'estado_civil_Separado', 'estado_civil_Solteiro',
       'estado_civil_União', 'qt_pessoas_residencia']).fillna(method='bfill').copy()
  y_2 = np.log(renda_encoded['renda'])
  X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, test_size=0.3, random_state=42)
  
  arvore_2 = DecisionTreeRegressor(max_depth=8 ,random_state= 42)
  arvore_2.fit(X_2_train, y_2_train)
  previsao_2 = arvore_2.predict(X_2_test)
  r_square_2 = r2_score(previsao_2, y_2_test)
  
  st.markdown(f'O valor de r-quadrado obtido dela foi {r_square_2:.3f}, com uma profundidade de {arvore_2.get_depth()}')
  
  
  plt.figure(figsize=(25, 10))
  plot_tree(arvore_2,
          filled=True,
          feature_names=X_2.columns)
  st.write('## Árvore de Decisão da base de dados tratada')
  st.pyplot(plt)
  st.markdown('------')
  
  if st.button("Salvar Imagem da Árvore de Decisão Tratada"):
      os.makedirs(output_folder_path, exist_ok=True)
        
      file_path = os.path.join(output_folder_path, "arvore_decisao_tratada.png")
      plt.savefig(file_path)
      st.success(f"Imagem salva em: {file_path}")
