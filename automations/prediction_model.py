
'''
Script python para predição do modelo XGBoost (personalizado com o limiar de decisão configurado para 0.25 
podendo mudar caso análises futuras comprovem um limiar melhor de acordo com novos dados).
--------------------------------------
Sabrina Otoni da Silva - 2024/01
'''
from pathlib import Path
import pickle

def predict(data, preprocessing_path=Path('../preprocessing'), model_path=Path('../model'), threshold=0.25):
    # Carregando os pickles de preprocessamento e modelo (XGBoostClassifier)
    with open(f'{preprocessing_path}/preprocessing.pkl', 'rb') as file:
        preprocessing_pipeline = pickle.load(file)

    with open(f'{model_path}/xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    # Aplicando a transformação no dado baseado no que já foi treinado anteriormente com os dados de treino
    data = preprocessing_pipeline.transform(data)
    
    # Predizendo através de probabilidade
    probas = xgb_model.predict_proba(data)[:, 1]
    
    # Verificando a probabilidade com o threshold estipulado para estipular a classe pertencente
    predictions = (probas >= threshold).astype(int)
    return predictions