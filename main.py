"""
@author: fred_
"""

import pandas as pd
import io
import requests
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB


def ordenar_por_coluna(df, coluna):
    df = df.sort_values(by=coluna)
    return df

def coleta_dados_casos():
    
    current_date = datetime.date.today() 
    year, week_num, day_of_week = current_date.isocalendar()

    geocode = '4102406'
    disease = 'dengue'
    format_file = 'csv'
    week_start = '1'
    #week_end = str(week_num)
    year_start = '2010'
    #year_end = str(year)


    #Filtro de consulta da API
    url = "https://info.dengue.mat.br/api/alertcity?geocode="+geocode+"&disease="+disease+"&format="+format_file+"&ew_start="+week_start+"&ew_end="+str(49)+"&ey_start="+year_start+"&ey_end="+str(2021)

    #Salvando consulta em um DataFrame
    s = requests.get(url).content
    df_casos = pd.read_csv(io.StringIO(s.decode('utf-8')))
    
    #Ordenando valores do Data Frame pela Semana Epdemiológica
    df_casos['data_iniSE'] = pd.to_datetime(df_casos['data_iniSE'])
    df_casos = ordenar_por_coluna(df_casos, 'SE')
    
    return df_casos

def coleta_dados_umidade():
    
    #Lendo os dados da umidade e armazenando em outro DataFrame
    df_umidade = pd.read_excel('umidade.xlsx')
    
    #Convertendo df_umidade['Data'] para o formato Date time
    df_umidade['Data'] = pd.to_datetime(df_umidade['Data'])
    df_umidade = ordenar_por_coluna(df_umidade, 'Data')

    return df_umidade

def coleta_dados_temperatura():
    #Lendo os dados da umidade e armazenando em outro DataFrame
    df_temperatura = pd.read_excel('temperatura.xlsx')
    df_temperatura['Data'] = pd.to_datetime(df_umidade['Data'])
    df_temperatura = ordenar_por_coluna(df_temperatura, 'Data')

    return df_temperatura

def excluir_colunas(df, colunas):
    df = df.drop(columns=colunas)
    return df

def renomear_colunas(df,colunas):
    df = df.rename(columns=colunas)
    return df

def preencher_faltantes_media(df,coluna):
    df[coluna] = df[coluna].fillna(df[coluna].mean())
    return df

def calcular_media_3_valores(df):
    for i in range(0,4306):
        if(pd.isna(df_umidade.media[i])):
            df_umidade.media[i] = (df_umidade.manha[i] + df_umidade.tarde[i] + df_umidade.noite[i] ) / 3
    
    return df

def calcular_media_2_valores(df):

    for i in range(0,4281):
        if(pd.isna(df_temperatura.media[i])):
            df_temperatura.media[i] = (df_temperatura.minima[i] + df_temperatura.maxima[i] ) / 2
    
    return df

def calcular_umidade_semanal():    
    medias_umidade_semanais = []
    semanas = []
    for i in range(0,623):
        if(i < 622):
            data_inicial = df_casos.iloc[i,0]
            data_final = df_casos.iloc[i+1,0]
            medias_temp = []
        else:
            data_inicial = df_casos.iloc[i,0]
            data_final = datetime.datetime(2021, 12, 15)
            medias_temp = []
        for j in range(0,4306):
            data_atual = df_umidade.iloc[j,0] 
            if(data_atual >= data_inicial and data_atual < data_final):
                medias_temp.append(df_umidade.iloc[j,1])
                j = j + 1
        if(len(medias_temp) == 0):
            i = i + 1   
        else:
            semanas.append(i+1)
            medias_umidade_semanais.append(sum(medias_temp)/len(medias_temp))
            medias_temp.clear()
            i = i + 1
        
    return medias_umidade_semanais
        
def calcular_temperatura_semanal():
    medias_temperatura_semanais = []
    semanas = []
    for i in range(0,623):
        if(i < 622):
            data_inicial = df_casos.iloc[i,0]
            data_final = df_casos.iloc[i+1,0]
            medias_temp_2 = []
        else:
            data_inicial = df_casos.iloc[i,0]
            data_final = datetime.datetime(2021, 12, 15)
            medias_temp_2 = []
        for j in range(0,4306):
            data_atual = df_temperatura.iloc[j,0] 
            if(data_atual >= data_inicial and data_atual < data_final):
                medias_temp_2.append(df_temperatura.iloc[j,1])
                j = j + 1
        if(len(medias_temp_2) == 0):
            i = i + 1    
        else:
            semanas.append(i+1)
            medias_temperatura_semanais.append(sum(medias_temp_2)/len(medias_temp_2))
            medias_temp_2.clear()
            i = i + 1
    
    return medias_temperatura_semanais

def classificacao_qtd_casos(df):
    if(df['Qtd_casos'] >= 0 and df['Qtd_casos'] <= 15):
        return 0
    elif(df['Qtd_casos'] > 15 and df['Qtd_casos'] < 40):
        return 1
    return 2

def treinar_knn():    
    n_splits = [5, 7, 9]
    n_repeats = [3, 6, 9]
    n_neighbors = [3, 6, 9, 11, 15]
    array_splits = []
    array_repeats = []
    array_acuracia = []
    array_neighboors = []
    
    for split in n_splits:
        for repeat in n_repeats:        
            for neigh in n_neighbors:    
                kfold = RepeatedKFold(n_splits=split, n_repeats=repeat)
                knn = KNeighborsClassifier(n_neighbors = neigh).fit(x,y)
                media_knn = cross_val_score(knn, x, y, scoring='accuracy', cv=kfold).mean()
                array_splits.append(split)
                array_repeats.append(repeat)
                array_neighboors.append(neigh)
                array_acuracia.append(media_knn)
    
    avaliacao_knn = pd.DataFrame(data={'Splits: ':array_splits,
                                       'Repetições: ':array_repeats,
                                       'Vizinhos: ':array_neighboors,
                                       'Acurácia: ':array_acuracia})
    return avaliacao_knn

def treinar_rede_neural():
    n_splits = [5, 7, 9]
    n_repeats = [3, 6, 9]
    max_iter =  [100, 300, 500]
    array_splits = []
    array_repeats = []
    array_acuracia = []
    array_max_iter = []
    
    for split in n_splits:
        for repeat in n_repeats:        
            for iteracao in max_iter: 
                kfold = RepeatedKFold(n_splits=split, n_repeats=repeat)
                clf = MLPClassifier(max_iter = iteracao).fit(x, y)
                media_clf = cross_val_score(clf, x, y, scoring='accuracy', cv=kfold).mean()
                array_splits.append(split)
                array_repeats.append(repeat)
                array_acuracia.append(media_clf)
                array_max_iter.append(iteracao)
    
    avaliacao_rede_neural = pd.DataFrame(data={'Splits: ':array_splits,
                                       'Repetições: ':array_repeats,
                                       'Iterações: ': array_max_iter,
                                       'Acurácia: ':array_acuracia})
    return avaliacao_rede_neural

def treinar_svc():    
    n_splits = [5, 7, 9]
    n_repeats = [3, 6, 9]
    array_splits = []
    array_repeats = []
    array_acuracia = []
    
    for split in n_splits:
        for repeat in n_repeats: 
            kfold = RepeatedKFold(n_splits=split, n_repeats=repeat)
            svc = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(x,y)
            media_svc = cross_val_score(svc, x, y, scoring='accuracy', cv=kfold).mean()
            array_splits.append(split)
            array_repeats.append(repeat)
            array_acuracia.append(media_svc)
            
    avaliacao_svc = pd.DataFrame(data={'Splits: ':array_splits,
                                       'Repetições: ':array_repeats,
                                       'Acurácia: ':array_acuracia})
    return avaliacao_svc

def treinar_linear_svc():
    n_splits = [5, 7, 9]
    n_repeats = [3, 6, 9]
    array_splits = []
    array_repeats = []
    array_acuracia = []
    
    for split in n_splits:
        for repeat in n_repeats: 
            kfold = RepeatedKFold(n_splits=split, n_repeats=repeat)
            Lsvc = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5)).fit(x,y)
            media_Lsvc = cross_val_score(Lsvc, x, y, scoring='accuracy', cv=kfold).mean()
            array_splits.append(split)
            array_repeats.append(repeat)
            array_acuracia.append(media_Lsvc)
            
    avaliacao_linear_svc = pd.DataFrame(data={'Splits: ':array_splits,
                                       'Repetições: ':array_repeats,
                                       'Acurácia: ':array_acuracia})
    return avaliacao_linear_svc

def treinar_sgdc():
    n_splits = [5, 7, 9]
    n_repeats = [3, 6, 9]
    max_iter =  [500, 1000, 1500]
    array_splits = []
    array_repeats = []
    array_acuracia = []
    array_max_iter = []
    
    for split in n_splits:
        for repeat in n_repeats:        
            for iteracao in max_iter: 
                kfold = RepeatedKFold(n_splits=split, n_repeats=repeat)
                sgdc = make_pipeline(StandardScaler(),SGDClassifier(max_iter=iteracao, tol=1e-3))
                media_sgdc = cross_val_score(sgdc, x, y, scoring='accuracy', cv=kfold).mean()
                array_splits.append(split)
                array_repeats.append(repeat)
                array_acuracia.append(media_sgdc)
                array_max_iter.append(iteracao)
    
    avaliacao_sgdc = pd.DataFrame(data={'Splits: ':array_splits,
                                       'Repetições: ':array_repeats,
                                       'Iterações: ': array_max_iter,
                                       'Acurácia: ':array_acuracia})
    return avaliacao_sgdc

def treinar_nb():
    n_splits = [5, 7, 9]
    n_repeats = [3, 6, 9]
    array_splits = []
    array_repeats = []
    array_acuracia = []
    
    for split in n_splits:
        for repeat in n_repeats: 
            kfold = RepeatedKFold(n_splits=split, n_repeats=repeat)
            nb = GaussianNB().fit(x,y)
            media_nb = cross_val_score(nb, x, y, scoring='accuracy', cv=kfold).mean()
            array_splits.append(split)
            array_repeats.append(repeat)
            array_acuracia.append(media_nb)
            
    avaliacao_nb = pd.DataFrame(data={'Splits: ':array_splits,
                                       'Repetições: ':array_repeats,
                                       'Acurácia: ':array_acuracia})
    return avaliacao_nb


####################################################################################################


df_casos = coleta_dados_casos()
df_umidade = coleta_dados_umidade()
df_temperatura = coleta_dados_temperatura()

colunas_casos=['casos_est','casos_est_min','casos_est_max','p_rt1','SE','p_inc100k','Localidade_id','id','nivel',
               'versao_modelo','tweet','Rt','pop','tempmin','umidmax','receptivo','transmissao','nivel_inc',
               'umidmed','umidmin','tempmed','tempmax','notif_accum_year'] 
colunas_umidade=['Id']
colunas_temperatura=['Id','rela']

df_casos = excluir_colunas(df_casos, colunas_casos)
df_umidade = excluir_colunas(df_umidade, colunas_umidade)
df_temperatura = excluir_colunas(df_temperatura, colunas_temperatura)

renomear_colunas_umidade={'09:00 (%)':'manha', 
                        '15:00 (%)':'tarde', 
                        '21:00 (%)':'noite',
                        'Média (%)' : 'media'}
renomear_colunas_temperatura={'tempmin':'minima',
                              'tempmedia':'media',
                              'tempmax':'maxima'}

df_umidade = renomear_colunas(df_umidade,renomear_colunas_umidade)
df_temperatura = renomear_colunas(df_temperatura, renomear_colunas_temperatura)

df_umidade = preencher_faltantes_media(df_umidade, 'manha')
df_umidade = preencher_faltantes_media(df_umidade, 'tarde')
df_umidade = preencher_faltantes_media(df_umidade, 'noite')

df_temperatura = preencher_faltantes_media(df_temperatura, 'minima')
df_temperatura = preencher_faltantes_media(df_temperatura, 'maxima')

df_umidade = calcular_media_3_valores(df_umidade)
df_temperatura = calcular_media_2_valores(df_temperatura)

df_umidade = excluir_colunas(df_umidade, ['manha','tarde','noite'] )

df_umidade = ordenar_por_coluna(df_umidade,['Data'])

umidades_semanais = calcular_umidade_semanal()
temperaturas_semanais = calcular_temperatura_semanal()

umidade_semanal = pd.DataFrame([umidades_semanais[::-1]])
umidade_semanal = umidade_semanal.transpose()

temperatura_semanal = pd.DataFrame([temperaturas_semanais[::-1]])
temperatura_semanal= temperatura_semanal.transpose()

frame = {'Data': df_casos['data_iniSE'],
         'Temp_media_sem': temperatura_semanal[0],
         'Umidade_media_sem': umidade_semanal[0],
         'Qtd_casos': df_casos['casos']}

df_casos_final = pd.DataFrame(frame)

df_casos_final = df_casos_final.drop(columns=['Data'])

df_casos_final = df_casos_final.drop(labels=0, axis=0)

df_casos_final['Risco'] = df_casos_final.apply(
                                lambda df : classificacao_qtd_casos(df), 
                                axis=1)

df_final = excluir_colunas(df_casos_final, 'Qtd_casos')


#Separando os dados entre variáveis independentes (x) e dependentes (y)
x = pd.DataFrame(df_final.iloc[:,0:2].values)

scaler = StandardScaler()
scaler.fit(x)

x = renomear_colunas(x, {0:'Temperatura',
                         1:'Umidade'})

y = df_final.iloc[:,2].values

##########################################

knn = treinar_knn()
linear_svc = treinar_linear_svc()
naive_bayes = treinar_nb()
rede_neural = treinar_rede_neural()
sgdc = treinar_sgdc()
svc = treinar_svc()




















