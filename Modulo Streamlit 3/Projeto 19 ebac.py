import pandas            as pd
import streamlit         as st
import seaborn           as sns
import matplotlib.pyplot as plt
from PIL                 import Image
from io                  import BytesIO
from openpyxl.workbook import Workbook
from sklearn           import metrics
import statsmodels.formula.api as smf
from scipy.stats import ks_2samp

st.set_page_config(page_title= 'Projeto Streamlit II',
                   page_icon= 'https://afubesp.org.br/wp-content/uploads/2022/07/logo_ebac-960x640.png',
                   layout= 'wide')

st.write('# Projeto Streamlit 2')
st.write('---------')
st.write('''Este é o projeto final para o módulo 19 do curso de ciência de dados da EBAC, aqui iremos mostrar os dados
         abordados ao longo deste módulo, as vezes de maneira similar ao que foi passado em aula, além das informações
         sobre a base de dados passada em aula, serão também algumas análises basícas sobre a base de dados em questão, como regressões logisticas e 
         seus valores de importância. para mais informações sobre a regressão, favor conslutar o arquivo "Projeto 19 (complemento).ipynb".''')

#Função para ler os dados (igual mostrada em aula)
@st.cache_data(show_spinner = True)
def load_data(data):
    try:
        return pd.read_csv(data, sep= ';')
    except:
        return pd.read_excel(data)
    
#Função para selecionar as colunas (igual em aula)
@st.cache_data()
def select_filter(relatorio, col, selected):
    if 'all' in selected:
        return relatorio
    else:
        return relatorio[relatorio[col].isin(selected)].reset_index(drop = True)
    
# Função para converter o df para csv(igual em aula)
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel(igual em aula)
@st.cache_data
def to_excel(df):
    output = BytesIO()
    df.to_excel(output, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def execute_analysis(bank, ages):
    bank = bank.query("age >= @ages[0] and age <= @ages[1]")
    bank = bank.rename(columns={'emp.var.rate':'emp_var_rate',
                'cons.price.idx': 'cons_price_idx',
                'cons.conf.idx': 'cons_conf_idx',
                'nr.employed':'nr_employed'})
    bank['y'] = bank['y'].replace({'yes': 1, 'no': 0})
    dummies = pd.get_dummies(bank[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']],
                             drop_first= True)
    bank = pd.concat([bank, dummies], axis=1)
    
    reg_pura = smf.logit("y ~  duration + pdays  + poutcome_success + emp_var_rate + euribor3m + nr_employed + poutcome_nonexistent + contact_telephone",
                         data= bank).fit()
    st.text(reg_pura.summary())
    
    bank['predito'] = reg_pura.predict()
    cat_pred = pd.qcut(bank['predito'], 5, duplicates= 'drop')
    group_reg = bank.groupby(cat_pred)
    qualid = group_reg[['predito']].count().rename(columns={'predito': 'contagem'})
    qualid['predito'] = group_reg['predito'].mean()
    qualid['pct_y'] = group_reg['y'].mean()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = qualid['pct_y'].plot(label='%Predito')
    ax = qualid['predito'].plot(label='%Observado')
    ticks = ax.set_xticks([0, 1, 2, 3, 4])
    labels = ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.legend(loc="lower right")
    ax.set_ylabel('Probabilidade de evento')
    ax.set_xlabel('Grupo')
    plt.title('Curva de calibragem')

    acc = metrics.accuracy_score(bank['y'], bank['predito'] > .5)
    fpr, tpr, thresholds = metrics.roc_curve(bank['y'], bank['predito'])
    auc_ = metrics.auc(fpr, tpr)
    gini = 2 * auc_ - 1
    ks = ks_2samp(bank.loc[bank['y'] == 1, 'predito'], bank.loc[bank['y'] != 1, 'predito']).statistic

    st.write(f'Acurácia: {(acc * 100):.2f}%\nGINI: {(gini * 100):.2f}%\nKS: {(ks * 100):.2f}%')


def main():
    #Criando a barra lateral e colocando a imagem
    st.sidebar.image ('https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png')
    #Criando o botão de subir os arquivos
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank marketing data", type = ['csv','xlsx'])
    
# Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        bank_raw = load_data(data_file_1)
        bank = bank_raw.copy()

        st.write('## Antes dos filtros')
        st.write(bank_raw.head())

        with st.sidebar.form(key='my_form'):

            # SELECIONA O TIPO DE GRÁFICO
            graph_type = st.radio('Tipo de gráfico:', ('Barras', 'Pizza'))
        
            # IDADES
            max_age = int(bank.age.max())
            min_age = int(bank.age.min())
            idades = st.slider(label='Idade', 
                                        min_value = min_age,
                                        max_value = max_age, 
                                        value = (min_age, max_age),
                                        step = 1)


            # PROFISSÕES
            jobs_list = bank.job.unique().tolist()
            jobs_list.append('all')
            jobs_selected =  st.multiselect("Profissão", jobs_list, ['all'])

            # ESTADO CIVIL
            marital_list = bank.marital.unique().tolist()
            marital_list.append('all')
            marital_selected =  st.multiselect("Estado civil", marital_list, ['all'])

            # DEFAULT?
            default_list = bank.default.unique().tolist()
            default_list.append('all')
            default_selected =  st.multiselect("Default", default_list, ['all'])

            
            # TEM FINANCIAMENTO IMOBILIÁRIO?
            housing_list = bank.housing.unique().tolist()
            housing_list.append('all')
            housing_selected =  st.multiselect("Tem financiamento imob?", housing_list, ['all'])

            
            # TEM EMPRÉSTIMO?
            loan_list = bank.loan.unique().tolist()
            loan_list.append('all')
            loan_selected =  st.multiselect("Tem empréstimo?", loan_list, ['all'])

            
            # MEIO DE CONTATO?
            contact_list = bank.contact.unique().tolist()
            contact_list.append('all')
            contact_selected =  st.multiselect("Meio de contato", contact_list, ['all'])

            
            # MÊS DO CONTATO
            month_list = bank.month.unique().tolist()
            month_list.append('all')
            month_selected =  st.multiselect("Mês do contato", month_list, ['all'])

            
            # DIA DA SEMANA
            day_of_week_list = bank.day_of_week.unique().tolist()
            day_of_week_list.append('all')
            day_of_week_selected =  st.multiselect("Dia da semana", day_of_week_list, ['all'])


                    
            # encadeamento de métodos para filtrar a seleção
            bank = (bank.query("age >= @idades[0] and age <= @idades[1]")
                        .pipe(select_filter, 'job', jobs_selected)
                        .pipe(select_filter, 'marital', marital_selected)
                        .pipe(select_filter, 'default', default_selected)
                        .pipe(select_filter, 'housing', housing_selected)
                        .pipe(select_filter, 'loan', loan_selected)
                        .pipe(select_filter, 'contact', contact_selected)
                        .pipe(select_filter, 'month', month_selected)
                        .pipe(select_filter, 'day_of_week', day_of_week_selected)
            )


            submit_button = st.form_submit_button(label='Aplicar')
        
        # Botões de download dos dados filtrados
        st.write('## Após os filtros')
        st.write(bank.head())
        
        df_xlsx = to_excel(bank)
        st.download_button(label='📥 Download tabela filtrada em EXCEL',
                            data=df_xlsx ,
                            file_name= 'bank_filtered.xlsx')
        st.markdown("---")

        # PLOTS    
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        bank_raw_target_perc = bank_raw.y.value_counts(normalize=True).sort_index() * 100

        try:
            bank_target_perc = bank.y.value_counts(normalize=True).sort_index() * 100
        except:
            st.error('Erro no filtro')
                
        # Botões de download dos dados dos gráficos
        col1, col2 = st.columns(2)

        df_xlsx = to_excel(bank_raw_target_perc.reset_index())
        col1.write('### Proporção original')
        col1.write(bank_raw_target_perc.reset_index())
        col1.download_button(label='📥 Download',
                            data=df_xlsx,
                            file_name='bank_raw_y.xlsx')

        df_xlsx = to_excel(bank_target_perc.reset_index())
        col2.write('### Proporção da tabela com filtros')
        col2.write(bank_target_perc.reset_index())
        col2.download_button(label='📥 Download',
                            data=df_xlsx,
                            file_name='bank_y.xlsx')

        st.markdown("---")

    st.write('## Proporção de aceite')
# PLOTS    
    if graph_type == 'Barras':
        sns.barplot(x=bank_raw_target_perc.index, y=bank_raw_target_perc, ax=ax[0])
        ax[0].set_title('Dados brutos', fontweight="bold")
        ax[0].bar_label(ax[0].containers[0], fmt='%.2f%%')

        sns.barplot(x=bank_target_perc.index, y=bank_target_perc, ax=ax[1])
        ax[1].set_title('Dados filtrados', fontweight="bold")
        ax[1].bar_label(ax[1].containers[0], fmt='%.2f%%')


    else:
        bank_raw_target_perc.plot(kind='pie', autopct='%.2f', ax=ax[0])
        ax[0].set_title('Dados brutos', fontweight="bold")

        bank_target_perc.plot(kind='pie', autopct='%.2f', ax=ax[1])
        ax[1].set_title('Dados filtrados', fontweight="bold")

    #Regressão Logistica
    if st.sidebar.checkbox("Executar análise"):
        execute_analysis(bank_raw, idades)
if __name__ == '__main__':
	main()
st.pyplot(plt)