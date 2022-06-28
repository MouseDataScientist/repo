import streamlit as st
import numpy as np
import os
import tensorflow as tf

st.header('Previsor de IDH')
st.markdown('Configure os parâmetros abaixo para efetuar a previsão')

ile = st.slider('Informe o Índice de Liberdade Econômica', min_value = 0.00, max_value = 10.00)
st.write(ile)

ipc = st.slider('Informe o Índice de Percepção da Corrupção', min_value = 0, max_value = 100)
st.write(ipc)

if (os.path.exists('modelo.h5')):
    modelo = tf.keras.models.load_model('modelo.h5')
    botao = st.button('PREVER')
    if (botao):
        listaValores = np.array([[ile, ipc]])
        previsao = modelo.predict(listaValores)
        if previsao[:, 0] >= 0.50:
            st.write('O IDH tende a ser ALTO!')
        elif previsao[:, 1] >=0.50:
            st.write('O IDH tende a ser BAIXO!')
        else:
            st.write('O IDH tende a ser MÉDIO!')
else:
    st.error('Erro ao carregar modelo preditivo, contate o administrador do sistema!')
