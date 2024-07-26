import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import tree
import statsmodels.api as sm
import matplotlib.pyplot as plt


@st.cache_data
def std_df(df):
    var_num = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay','OperatingSystems', 'Browser', 'Region', 'TrafficType']

    var_cat = ['VisitorType','Weekend', 'Revenue']
    columns_delete = ['ExitRates', 'PageValues','OperatingSystems', 'Browser','TrafficType','Region', 'TrafficType', 'Weekend']
    
    df_pad = pd.DataFrame(StandardScaler().fit_transform(df[var_num]), columns= df[var_num].columns)
    df_pad = (pd.concat([df_pad, pd.get_dummies(df[var_cat],drop_first= True)], axis=1)
              .drop(columns= columns_delete))
    
    return df_pad

@st.cache_data
def print_elbow(df):
    df_pad = std_df(df)
    SSD = []

    for i in range (1, 10):
        km = KMeans(n_clusters= i, n_init=10)
        km = km.fit(df_pad)
        SSD.append(km.inertia_)
         
    db = pd.DataFrame({'num_clusters': list(range(1, len(SSD)+1)), 'SSD': SSD})
    fig, ax = plt.subplots(figsize = (12, 4))
    db.plot(x='num_clusters', y='SSD', ax=ax)
    
    return fig

@st.cache_data
def cluster(df):
    df_pad = std_df(df)
    cluster = AgglomerativeClustering(linkage='complete',
                                  n_clusters=4,
                                  distance_threshold= None).fit(df_pad)
    df_pad['grupo'] = pd.Categorical(cluster.labels_)
    return df_pad

@st.cache_data
def tree(df):
    df_pad = cluster(df)
    # Identificar e converter colunas booleanas
    boolean_columns = df_pad.select_dtypes(include='bool').columns
    df_pad[boolean_columns] = df_pad[boolean_columns].astype(int)

    X = df_pad.drop(columns='grupo')
    y = df_pad['grupo']
    
    model = sm.MNLogit(y, X).fit()
    
    return model

    
# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title='Análise Clusters',
                       layout="wide",
                       initial_sidebar_state='expanded',
                       page_icon= 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg',)

    # Título principal da aplicação
    st.write("""# Clusterização e análise de grupos 📶

    Clusterizar um conjunto de dados significa dividir este conjunto em diversos subconjuntos diferentes, de modo
    que essa divisão é feita com base em uma das variáveis desse banco de dados. Os conjuntos são criados de tal maneira
    que os indivíduos semelhantes entre si ficam dentro de um mesmo conjunto, e estes conjuntos são diferentes um do outro

    Neste caso iremos analisar uma tabela com diversas informações sobre clientes e se foi feita alguma compra ou não, com isso, 
    iremos agrupar estes clientes em grupos distintos de compradores e análisar como estes grupos são compostos,
    para isso, temos a base de dados abaixo:
    """)
    st.markdown("---")
    
    
    st.write('Segue abaixo uma descrição da base de dados e oque cada coluna significa. 📜')
    st.markdown('''|Variavel                |Descrição          | 
|------------------------|:-------------------| 
|Administrative          | Quantidade de acessos em páginas administrativas| 
|Administrative_Duration | Tempo de acesso em páginas administrativas | 
|Informational           | Quantidade de acessos em páginas informativas  | 
|Informational_Duration  | Tempo de acesso em páginas informativas  | 
|ProductRelated          | Quantidade de acessos em páginas de produtos | 
|ProductRelated_Duration | Tempo de acesso em páginas de produtos | 
|BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | 
|ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | 
|PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | 
|SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | 
|Month                   | Mês  | 
|OperatingSystems        | Sistema operacional do visitante | 
|Browser                 | Browser do visitante | 
|Region                  | Região | 
|TrafficType             | Tipo de tráfego                  | 
|VisitorType             | Tipo de visitante: novo ou recorrente | 
|Weekend                 | Indica final de semana | 
|Revenue                 | Indica se houve compra ou não |''')
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dataframe" , "💻 Clusterização" , "📊 Tratamentos" , "💡 Análise Insights", "🏁 Conclusões"])
    st.markdown("---")
    # URL do arquivo
    data_url = 'https://raw.githubusercontent.com/ThiagoVieira96/Projetos-Ebac/main/Projeto_clusters/online_shoppers_intention.csv'

    # Carregar dados diretamente do URL
    df_compras = pd.read_csv(data_url, infer_datetime_format=True)
    
    # Conteúdo da barra lateral
    st.sidebar.image('https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png')
    st.sidebar.header("Projeto Cluster 📊")
    st.sidebar.write("""
    Esta aplicação realiza clusterização de clientes pelo perfil de compra, além da análise destes clusters.
    
    - O arquivo de dados utilizado pode ser encontrado [aqui](https://raw.githubusercontent.com/ThiagoVieira96/Projetos-Ebac/main/Projeto_clusters/online_shoppers_intention.csv).
    
    - O repositório com o código fonte desta aplicação pode ser encontrado [aqui](https://github.com/ThiagoVieira96/Projetos-Ebac/tree/main/Projeto_clusters)
    
    Informações adicionais 💡:
    - O gráfico do método do cotovelo foi feito usando clusterização por KMeans, uma técnica bastante simples e popular em análise de dados.
    - As regressões foram feitas usando o método MNLogit do statsmodels, isso é, uma regressão logistica multivariada.
    """)    
    
    with tab1:
        st.write(''' ## Base de dados 📂
                 
        Aqui iremos basicamente visualizar a base de dados sem nenhum tratamento, basicamente para vermos que tipos de dados
     esta base possui, assim como que tipos de tratamentos teremos que realizar 
                 ''')
        
        
        st.write(df_compras.head(10))  
        
        
        col1, col2 = st.columns(2)

# Adicionando caixas de texto nas colunas
        with col1:
            col_1, col_2 = st.columns(2)
            with col_1:
                st.write("Valores Missing",df_compras.isna().sum())
            with col_2:
                st.write("Variáveis",df_compras.dtypes)
            

        with col2:
            st.write('Análise das variáveis:')
            st.write('''\nPela tabela ao lado, vemos que as variáveis categoricas são apenas "Month" e "VisitoType",
                     juntamente com "Weekend" e "Revenue" que são booleanas. Além disso, como vimos também, não existem colunas faltando neste
                     banco de dados, assim, vamos prosseguir para a padronização e o tratamento das variáveis,
                     em seguida iremos para a divisão entre os diferentes tipos de clientes, por fim iremos
                     obter insights a partir de análises estatisticas sobre os grupos.''')
            
    with tab2:
        st.write(''' ## Clusterização   💻
                 
                 Agora iremos começar a parte de tratamentos na nossa base de dados, primeiramente vamos separar a base de dados
     nas variáveis numericas e não numéricas.
     Após isso, iremos padronizar as variáveis numéricas e criar dummies para as categoricas, para isso, iremos assumir o seguinte:
     - O mês não interfere na chance de uma pessoa realizar uma compra ou não. Isso nos possibilitará remover a variável "Month",
     o que geraria doze variáveis dummy, algo que iria interferir muito na complexidade do modelo, mesmo sabendo que isso
     não é interiramente verdade.
     Para contornar isso, iremos deixar a coluna "SpecialDay", que mostra a distancia de uma data festiva,
     que também interfere na possibilidade de compra.
     Isso irá nos poupar de fazer uma análise muito complexa e computacionalmente custosa.''')
        
        st.write('''A primeira parte dos tratamentos será dividir os grupos em clusters diferentes, ou seja, grupo distintos entre si,
                 porém os membros de um mesmo grupo são semelhantes. Para isso, o primeiro passo será definir o número de grupos a ser divididos,
                 para isso, usamos uma técnica chamada método do cotovelo, isso é, quando houver uma queda brusca nos valores de y em relação a x,
                 teremos a quantidade ideal de clusters.''')
        
        st.pyplot(print_elbow(df_compras))
        st.write('''Com base no gráfico acima, iremos escolher dividir os clientes em 4 grupos diferentes,
                 este número não somente é um bom número, como mostrado pelo gráfico,
                 mas também não é um número muito grande, a ponto da nossa análise ser afetada.''')
    with tab3:
        st.write('''#### Agora, com os grupos definidos, podemos rodar uma árvore de regressão para identificar as variáveis mais influentes para a nossa variável resposta. com isso, poderemos identificar o perfil dos nossos clientes. ✅''')
        tab_sum_1, tab_sum_2 = st.tabs(['Análise três primeiros grupos', 'Análise três últimos grupos'])
        
        with tab_sum_1:
            st.text(tree(df_compras).summary2())
            
        with tab_sum_2:
            st.text(tree(df_compras).summary())
        
        st.write('''Como vemos acima, cada um dos grupos possui variáveis mais importantes para seu resultado e outras menos,
                 iremos aceitar em cada grupo as variáveis com o valor de "P > |t|" menor que 0,05. Com isso, as variáveis aceitas
                 de cada grupo serão: ''')
        
        st.write('- Grupo 0: Administrative, Informational,  Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, SpecialDay, VisitorType_Returning_Visitor.')
        st.write('- Grupo 1: Administrative, Informational,  Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, SpecialDay, VisitorType_Returning_Visitor. ')
        st.write('- Grupo 2: Informational_Duration, VisitorType_Returning_Visitor.')
        st.write('- Grupo 4: Administrative_Duration, Informational, ProductRelated, ProductRelated_Duration, BounceRates, VisitorType_Returning_Visitor.')
        st.write('Podemos criar grupamentos entre estes grupos e as variáveis mais importantes para eles, assim, poderemos descobrir o perfil de cada grupo.')
        
    with tab4:
        df = pd.concat([df_compras, cluster(df_compras)['grupo']], axis= 1)
        
        st.write("## Estatísticas por Grupo 📑")
        
        # Obter todos os valores únicos da coluna 'grupo'
        grupos = df['grupo'].sort_values().unique()
        
        # Iterar sobre cada grupo e calcular estatísticas descritivas
        for grupo in grupos:
            st.write(f"### Estatísticas para o Grupo {grupo}")
            
            # Filtrar o DataFrame para o grupo atual
            df_grupo = df[df['grupo'] == grupo]
            
            if grupo == 0:
                stats = df_grupo[['Administrative', 'Informational',
                              'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                              'BounceRates', 'SpecialDay']].describe()
                st.write('''Como podemos ver, o grupo zero é um grupo de outliers, contendo apenas 3 membros, que provavelmente seriam agrupados
                         juntamente ao grupo 1, caso o número de clusters fosse menor. Ao analisar as variáveis mais importantes para estes,
                         vemos que os membros deste grupo se destacam por uma grande quantidade de acessos em páginas administrativas, informativas
                         e páginas de produtos, além de passar muito tempo nestas páginas.''')
            elif grupo == 1:
                stats = df_grupo[['Administrative', 'Informational',
                              'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                              'BounceRates', 'SpecialDay']].describe()
                st.write('''O grupo 1, como vemos, é de longe o grupo mais numeroso de todos, com mais de 12000 membros. Mediante análise de suas
                         variáveis mais importantes, vemos que este grupo se destaca pela quantidade relativamente pequena de acessos em
                         páginas administrativas e informativas, além do pouco tempo passado nestas páginas, já a quantidade de acessos e de tempo passado
                         nestas páginas é relativamente maior. Outro ponto de interesse é que uma quantidade apreciável destes membros acessaram estas
                         páginas relativamente próximos de dias festivos, além disso, o Bounce Rate, isto é, o percentual
                        de clientes que acessaram os sites e saíram sem acionar quaiquer outros requests é relativamente alto.''')
            elif grupo == 2:
                stats = df_grupo[['Informational_Duration']].describe()
                st.write('''Sobre o grupo 2, a única variável estatisticamente relevante para este grupo é a variável "Informational_Duration"
                         , como só temos essa variável para a análise, vemos que os membros deste grupo passam uma quantidade de tempo em páginas
                         informativas que é intermediaria entre os grupos 0 e 1, além disso, existe um grande desvio padrão neste grupo, sendo assim,
                         o tempo que estes usuários passam nestas páginas varia bastante.''')
                
            else:
                stats = df_grupo[['Administrative_Duration', 'Informational', 'ProductRelated', 'ProductRelated_Duration',
                              'BounceRates']].describe()
                st.write('''Por fim, o grupo 3 se destaca pela grande quantidade de tempo passado em páginas administrativas e principalmente em
                         páginas relacionadas à produtos, sendo esta uma das maiores médias em todos os grupos. Além disso, a quantidade de acessos
                         em páginas informacionais e páginas relacionadas à produtos também é grande, ainda que não tão grande quanto o grupo 0, porém,
                         como o grupo 0 é composto apenas por outliers, podemos afirmar que a quantidade de tempo passada pelo grupo 3 nessas páginas
                         é, em média, a maior entre os grupo.''')
            st.write(stats)
    with tab5:
        st.write('### Com base nas análises feitas, podemos chegar às seguintes conclusões:')
        st.write('''
                 - As variáveis mais importantes para separar os grupos são o tempo de uso e a quantidade de acessos em sites informativos, administrativos
                 e sites de produtos, isso é verdade para todos os grupos.
                 - O grupo 0 é claramente um grupo de outliers do grupo 1, o mais populoso deles, isso somente ocorreu pois o número de clusters foi predefinido
                 para 4. Caso fossem feitos 3 clusters os membros do grupo 0 teriam sido agrupados juntamente com o grupo 1.
                 - Justamente por ser tão populoso, é perfeitamente possível que existam outros subgrupos dentro do grupo 1, caso sejam feitos mais clusters
                 estes subgrupos poderiam compôr um grupo separado.
                 - O fato do grupo 1 ser tão populoso indica que a maior parte dos clientes possui tendências semelhantes.
                 ''')
        
if __name__ == '__main__':
    main()
