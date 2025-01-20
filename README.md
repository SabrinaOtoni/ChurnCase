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
## Reproduzir o projeto localmente
Para garantir que o projeto seja reproduzido com as mesmas versões de bibliotecas utilizadas no desenvolvimento, utilizamos a ferramenta [UV](https://docs.astral.sh/uv/). 
O UV gerencia dependências de forma simples e rápida, garantindo um ambiente 100% reprodutível.

### Pré-requisitos
1. **Python 3.10 ou superior** (o projeto foi testado com Python 3.11).
2. **pip** instalado.
3. **Git** (caso deseje clonar o repositório via terminal).
Executar os comandos abaixo usando git bash.

### Passo 1: Clonar o repositório
Clone o repositório diretamente do GitHub:
```bash
git clone https://github.com/SabrinaOtoni/ChurnCase.git
```

### Passo 2: Instalar a UV
No terminal, instale a UV com o `pip`:
```bash
pip install uv
```

### Passo 3: Sincronizar o ambiente
No diretório do projeto (onde está o `README.md`), rode:
```bash
uv sync
source .venv/Scripts/activate
```
Isso vai criar (ou atualizar) um ambiente virtual .venv, instalará todas as bibliotecas listadas nos arquivos pyproject.toml e uv.lock, e por fim, ativar o ambiente virtual.
As versões das bibliotecas serão exatamente as mesmas utilizadas no desenvolvimento.

### Passo 4: Executar notebooks ou scripts
- **Abrir os notebooks**:
  ```bash
  uv run jupyter notebook
  ```
  Em seguida, navegue até a pasta `notebooks` e abra o arquivo desejado.

- **Executar o frontend**:
  Se quiser testar o modelo, rode o comando abaixo:
  ```bash
  uv exec -- streamlit run frontend_streamlit.py
  ```
  
## Linkedin
https://www.linkedin.com/in/sabrina-otoni-da-silva-22525519b/
