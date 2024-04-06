import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os

sns.set()  

#O projeto final usará o mesmo código usado em aula, iremos apenas adicionar funcinalidades.

def plota_pivot_table(df, value, index, func, ylabel, xlabel=None, opcao='nada'):
    print(f"Debug: value={value}, index={index}, func={func}, ylabel={ylabel}, xlabel={xlabel}, opcao={opcao}")
    if opcao == 'nada':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).plot(figsize=[15, 5])
    elif opcao == 'unstack':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).unstack().plot(figsize=[15, 5])
    elif opcao == 'sort':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).sort_values(value).plot(figsize=[15, 5])
    plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    st.pyplot(fig=plt)
    return None

st.set_page_config(page_title = 'SINASC Rondônia',
                    page_icon='https://upload.wikimedia.org/wikipedia/commons/e/ea/Flag_map_of_Rondonia.png',
                    layout='wide')

st.write('# Análise SINASC')
st.write('Este é o projeto final para o primeiro módulo de streamlit, os gráficos usados aqui são os mesmos usados nas análises feitas em aula, iremos apenas adicionar funcionalidades.')




sinasc = pd.read_csv('https://raw.githubusercontent.com/ThiagoVieira96/Projetos-Ebac/main/Modulo%20Streamlit%201/input_M15_SINASC_RO_2019.csv')

sinasc.DTNASC = pd.to_datetime(sinasc.DTNASC)

min_data = sinasc.DTNASC.min()
max_data = sinasc.DTNASC.max()

st.sidebar.write('# Menu')

data_inicial = st.sidebar.date_input('Data inicial', 
                value = min_data,
                min_value = min_data,
                max_value = max_data)
data_final = st.sidebar.date_input('Data inicial', 
                value = max_data,
                min_value = min_data,
                max_value = max_data)    

st.write(f'Período das análises:De: __{data_inicial}__. Até: __{data_final}__')
st.write('Os gráficos desejados irão aparecer abaixo.')

sinasc  = sinasc[(sinasc['DTNASC'] <= pd.to_datetime(data_final)) & (sinasc['DTNASC'] >=pd.to_datetime(data_inicial) )]

#Iremos mostrar cada gráfico separadamente, e escolher quais serão mostrados por vez.
chart_functions = {
    'Média da idade da mãe por data': ('IDADEMAE', 'DTNASC', 'mean', 'média idade mãe por data', 'data nascimento', 'nada'),
    'Média da idade da mãe por data e sexo': ('IDADEMAE', ['DTNASC', 'SEXO'], 'mean', 'media idade mae', 'data de nascimento', 'unstack'),
    'Peso médio do bebê': ('PESO', ['DTNASC', 'SEXO'], 'mean', 'media peso bebe', 'data de nascimento', 'unstack'),
    'Peso mediano por escolaridade da mãe': ('PESO', 'ESCMAE', 'median', 'PESO mediano', 'escolaridade mae', 'sort'),
    'APGAR 1': ('APGAR1', 'GESTACAO', 'mean', 'apgar1 medio', 'gestacao', 'sort')
}

selected_charts = st.sidebar.multiselect('Selecione os gráficos desejados:', list(chart_functions.keys()))

for chart in selected_charts:
    plota_pivot_table(sinasc, *chart_functions[chart])

def salvar_graficos(selected_charts, chart_functions, sinasc, caminho_salvamento):
    if not os.path.exists(caminho_salvamento):
        os.makedirs(caminho_salvamento)
    for chart in selected_charts:
        params = chart_functions[chart]
        xlabel = params[4]
        fig, ax = plt.subplots(figsize=[15, 5])
        plota_pivot_table(sinasc, *params[:-1], opcao=params[-1])
        plt.title(chart)
        
        # Cria o caminho completo do arquivo
        nome_arquivo = f"{chart.replace(' ', '_').lower()}.png"
        caminho_completo = os.path.join(caminho_salvamento, nome_arquivo)
        
        # Salva a figura no caminho especificado
        plt.savefig(caminho_completo)
        plt.close(fig)
        

# Caminho de salvamento local
caminho_salvamento_local = "gráficos"

# Botão para salvar os gráficos
if st.sidebar.button("Salvar Gráficos"):
    salvar_graficos(selected_charts, chart_functions, sinasc, caminho_salvamento_local)
    st.success("Gráficos salvos com sucesso no caminho local!")
