# Cancelamento de Clientes

## Descrição
Esse projeto foca na análise de um conjunto de dados fornecido pela IBM para demonstração da ferramenta IBM Cognos Analytics. O dataset apresenta informações detalhadas sobre 7043 clientes de uma empresa fictícia de telecomunicações da Califórnia, abrangendo serviços de telefonia residencial e internet no 3º trimestre.

## Objetivo
O objetivo principal desse projeto é aplicar e aprimorar habilidades em ciência de dados e machine learning. Através de um pré tratamento dos dados, análise exploratória, técnicas de data splitting, feature engineering, implementação de modelos de machine learning, utilização de pipelines e fine tunning, o projeto busca identificar padrões e fatores que contribuem para o cancelamento (churn) de clientes. Esse é um projeto de estudo, visando desenvolver competências práticas e teóricas.

## Base de dados:
https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset

## Motivação
Como iniciante na área de ciência de dados, estou buscando aprimorar meus conhecimentos e habilidades práticas. Esse projeto serve como um campo de treinamento para explorar e aplicar conceitos teóricos em um cenário realista, proporcionando uma experiência valiosa em solução de problemas e tomada de decisão baseada em dados.

## Estrutura
O projeto foi organizado de uma forma linear de modo a facilitar a navegação e compreensão do passo a passo.
```
├── automations
│   │   ├── data_processing.py
│   │   ├── folders_creation.py
│   │   ├── frontend_streamlit.py
│   │   ├── prediction_model.py
├── data
│   ├── d01_raw
│   │   ├── telco_customer_churn.xlsx
│   │   ├── text.csv
│   ├── d02_intermediate
│   ├── ├── X_test.csv
│   ├── ├── X_train.csv
│   ├── ├── telco_customer_churn_v2.xlsx
│   ├── ├── y_test.csv
│   ├── ├── y_train.csv
│   ├── d04_models
│   ├── ├── score_models.csv
├── notebooks
│   ├── n00_data_preparation.ipynb
│   ├── n01_exploratory_data_analysis.ipynb
│   ├── n02_data_split.ipynb
│   ├── n03_feature_engineering.ipynb
│   ├── n04_check_combinations.ipynb
│   ├── n05_test_model.ipynb
│   ├── n06_model_validation.ipynb
├── preprocessing
│   ├── boxcox_transformer_model.pkl
│   ├── kmeans_model.pkl
│   ├── log_transformer_model.pkl
│   ├── preprocessing.pkl
│   ├── rbf_transformer_model.pkl
├── README.md
└── requirements.txt
```

## Linkedin
https://www.linkedin.com/in/sabrina-otoni-da-silva-22525519b/

## Agradecimentos
