
'''
Front-end feito em python utilizando o framework Streamlit com o objetivo de utilizar o modelo para predições sob o case de Churn.
--------------------------------------
Sabrina Otoni da Silva - 2024/01
'''
from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
import data_processing

# Função para carregar o modelo e o pré-processador
def load_model_preprocessor():
    current_dir = Path(__file__).parent
    model_path = current_dir / Path('../model/xgb_model.pkl')
    preprocessor_path = current_dir / Path('../preprocessing/preprocessing.pkl')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    with open(preprocessor_path, 'rb') as file:
        preprocessor = pickle.load(file)
    return model, preprocessor

# Função para validar os dados de entrada
def validate_input(data):
    try:
        data['Latitude'] = float(data['Latitude'])
        data['Longitude'] = float(data['Longitude'])
        data['Tenure Months'] = int(data['Tenure Months'])
        data['Monthly Charges'] = float(data['Monthly Charges'])
        data['Total Charges'] = float(data['Total Charges'])
        return True, ""
    except ValueError as e:
        return False, str(e)

# Função para prever a probabilidade de churn
def predict_churn(model, preprocessor, data):
    df = pd.DataFrame([data])
    df = preprocessor.transform(df)
    proba = model.predict_proba(df)[0][1]
    return proba

# Carregando o modelo e o pré-processador
model, preprocessor = load_model_preprocessor()

# Definindo a interface do usuário no Streamlit
st.title('Bem vindo(a) ao sistema de controle dos clientes da empresa Telco!')
st.write('Aqui você consegue ter uma visibilidade (probabilística) se um cliente vai cancelar com a companhia de acordo com suas características. \n\n')
st.title('Nosso Serviço:\n')
st.write('- Previsão de Cancelamento (propensão de churn): Utilizando aprendizado de máquina, oferecemos probabilidade de um cliente cancelar os serviços de acordo com o perfil oferecido para ajudá-lo a gerenciar riscos e tomar decisões informadas.')

# Campos de entrada
input_data = {}

input_data['City'] = st.text_input('City')
input_data['Latitude'] = st.text_input('Latitude')
input_data['Longitude'] = st.text_input('Longitude')
input_data['Gender'] = st.selectbox('Gender', ['Male', 'Female'])
input_data['Senior Citizen'] = st.selectbox('Senior Citizen', ['No', 'Yes'])
input_data['Partner'] = st.selectbox('Partner', ['No', 'Yes'])
input_data['Dependents'] = st.selectbox('Dependents', ['No', 'Yes'])
input_data['Tenure Months'] = st.text_input('Tenure Months')
input_data['Phone Service'] = st.selectbox('Phone Service', ['No', 'Yes'])
input_data['Multiple Lines'] = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
input_data['Internet Service'] = st.selectbox('Internet Service', ['No', 'DSL', 'Fiber optic'])
input_data['Online Security'] = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
input_data['Online Backup'] = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
input_data['Device Protection'] = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
input_data['Tech Support'] = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
input_data['Streaming TV'] = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
input_data['Streaming Movies'] = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
input_data['Contract'] = st.selectbox('Contract', ['Two year', 'Month-to-month', 'One year'])
input_data['Paperless Billing'] = st.selectbox('Paperless Billing', ['No', 'Yes'])
input_data['Payment Method'] = st.selectbox('Payment Method', ['Bank transfer (automatic)', 'Electronic check', 'Credit card (automatic)', 'Mailed check'])
input_data['Monthly Charges'] = st.text_input('Monthly Charges')
input_data['Total Charges'] = st.text_input('Total Charges')

# Botão de previsão
if st.button('Prever Cancelamento'):
    valid, message = validate_input(input_data)
    if valid:
        proba = predict_churn(model, preprocessor, input_data)
        st.write(f'A probabilidade desse cliente cancelar é de: {proba:.2%}')
    else:
        st.error(f'Erro na entrada de dados: {message}')

# Verificando se todos os campos foram preenchidos
if not all(input_data.values()):
    st.warning('Por favor, preencha todas as informações necessárias.')
