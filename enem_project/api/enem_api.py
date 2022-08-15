#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from flask import Flask, render_template, request

# In[2]:


file = open('H:/RAFAEL/Trabalho/Portifólio/enem_project/model/modelo.pkl', 'rb')
modelo = pickle.load(file)


# In[3]:


app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    data_json = request.get_json()
    
    if data_json:
        if isinstance(data_json, dict):
            to_dataframe = pd.DataFrame(data_json, index = [0])
        else:
            to_dataframe = pd.DataFrame(data_json, columns = data_json[0].keys())
            
    predict = modelo.predict(to_dataframe)
    to_dataframe['prediction'] = predict
    return to_dataframe.to_json(orient='records')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port ='5000')