import streamlit as st
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.header('Previsor de IDH')
st.markdown('1º Configure os parâmetros abaixo')
st.markdown('2º Clieque em "PREVER" para efetuar a previsão')

ile = st.slider('Informe o Índice de Liberdade Econômica', min_value = 0.00, max_value = 10.00)
st.write(ile)

ipc = st.slider('Informe o Índice de Percepção da Corrupção', min_value = 0, max_value = 100)
st.write(ipc)

def prever(x):
    previsao = modelo.predict(x)
    if previsao[:,0] > 0.50:
        st.write('O IDH tende a ser ALTO!')
    elif previsao[:,1] > 0.50:
        st.write('O IDH tende a ser BAIXO!')
    else:
        st.write('O IDH tende a ser MÉDIO!')
        
normalizar = MinMaxScaler(feature_range = (0,1))

if ((os.path.exists('idh_project/modelo.h5')) and (os.path.exists('idh_project/previsores.npy')) and (os.path.exists('idh_project/normalizados.npy'))):
    modelo = tf.keras.models.load_model('idh_project/modelo.h5')
    previsores = np.load('idh_project/previsores.npy')
    normalizados = np.load('idh_project/normalizados.npy')
    botao = st.button('PREVER')
    
    if (botao):
        for i in range(len(previsores)):
            cont = i + 1
            novo = [[ile, ipc]]
            
            if np.all(novo[0] == previsores[i]):
                x = np.array([normalizados[i]])
                prever(x)
                break
    
            else:
                cont == len(previsores)
                previsores = np.append(previsores, novo, axis = 0)
                normalizados = normalizar.fit_transform(previsores)
                x = np.array([normalizados[-1]])
                prever(x)
                break
else:
    st.error('Erro ao carregar arquivos, contate o administrador do sistema!')
