import numpy as np
import streamlit as stl
import os
import keras

stl.header('Previsor de Índice de Desenvolvimento Humano')
stl.subheader('Modelo desenvolvido em Python com utilização de Redes Neurais')
stl.markdown('A taxa de acerto nas previsões altualmente está em 70%')

ile = stl.slider('Informe o Índice de Liberdade Econômica', min_value = 0.00, max_value = 10.00)
stl.write(ile)

ipc = stl.slider('Informe o Índice de Percepção da Corrupção', min_value = 0, max_value = 100)
stl.write(ipc)

if (os.path.exists('modelo.h5')):
    modelo = keras.models.load_model('modelo.h5')
    botao = stl.button('PREVER')
    if (botao):
        listaValores = np.array([[ile, ipc]])
        previsao = modelo.predict(listaValores)
        if previsao[:, 0] >= 0.50:
            stl.write('O IDH tende a ser ALTO!')
        elif previsao[:, 1] >=0.50:
            stl.write('O IDH tende a ser BAIXO!')
        else:
            stl.write('O IDH tende a ser MÉDIO!')
else:
    stl.error('Erro ao carregar modelo preditivo, contate o administrador do sistema!')