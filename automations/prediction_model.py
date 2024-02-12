
'''
Script python para predição do modelo XGBoost (personalizado com o limiar de decisão configurado para 0.25 
podendo mudar caso análises futuras comprovem um limiar melhor de acordo com novos dados).
--------------------------------------
Sabrina Otoni da Silva - 2024/01
'''
from pathlib import Path
import pickle

def predict(data, preprocessing_path=Path('../preprocessing'), model_path=Path('../model'), threshold=0.25):

    with open(f'{preprocessing_path}/preprocessing.pkl', 'rb') as file:
        preprocessing_pipeline = pickle.load(file)

    with open(f'{model_path}/xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    data = preprocessing_pipeline.transform(data)
    
    probas = xgb_model.predict_proba(data)[:, 1]
    
    predictions = (probas >= threshold).astype(int)
    return predictions