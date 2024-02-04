
'''
Script python para tratamento do dataset "telco_customer_churn" disponibilizado pela IBM para demonstração da ferramenta IBM Cognos Analytics.

Sabrina Otoni da Silva - 2024/01
'''

# Importação das bibliotecas necessárias.
import os
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.signal import find_peaks

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

import pickle

class DataTreat(BaseEstimator, TransformerMixin):
    """
    Um transformador que aplica tratamentos específicos a um DataFrame.

    Métodos:
    fit: Não faz nenhuma operação de ajuste, retorna o próprio objeto.
    transform: Aplica transformações especificadas em um DataFrame.

    Levanta:
    ValueError: Se a validação específica falhar.
    """

    def fit(self, X, y=None):
        """
        Método de ajuste. Não realiza nenhuma operação.
        Retorna a própria instância da classe.
        """
        return self

    def transform(self, df):
        """
        Aplica transformações específicas ao DataFrame.
        Retorna o DataFrame tratado.
        """
        l1 = [len(str(i).split()) for i in df['Total Charges']]
        l2 = [i for i in range(len(l1)) if l1[i] != 1]

        if not (df.loc[df['Tenure Months'] == 0].index == l2).all():
            print('False')
            raise ValueError("A validação falhou, o processamento não pode continuar.") 
        
        else:
            print('True')

        for i in l2:
            df.loc[i, 'Total Charges'] = df.loc[i, 'Monthly Charges'].astype(str)

        df['Total Charges'] = df['Total Charges'].astype(np.float64)
        return df
    
class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Um transformador que aplica a transformação logarítmica a colunas específicas de um DataFrame.
    Essa classe pode ser inicializada com um caminho para um modelo treinado salvo como um arquivo pickle,
    ou pode ser treinada com um novo conjunto de dados.

    Parâmetros:
    columns: list of str, opcional
        Nomes das colunas no DataFrame a serem transformadas.
    model_path: str, opcional
        Caminho para um arquivo pickle contendo um modelo de transformação logarítmica pré-treinado.
    """
    def __init__(self, columns=None, model_path=None):
        self.columns = columns
        self.model_path = model_path

    def fit(self, X, y=None):
        """
        Método de ajuste que define as colunas para transformação.
        """
        if self.model_path is not None:
            with open(self.model_path, 'rb') as file:
                self.columns = pickle.load(file)

        if self.columns is None:
            raise ValueError("Colunas não foram especificadas. Você deve especificar as colunas.")
        
        if self.model_path is None:
            # Verificar se as colunas especificadas estão presentes no DataFrame
            missing_cols = [col for col in self.columns if col not in X.columns]
            
            if missing_cols:
                raise ValueError(f"As colunas {missing_cols} não estão presentes no DataFrame.")
            # Salvar as colunas em um arquivo pickle
            
            preprocesspath = Path('../preprocessing')
            
            with open(f'{preprocesspath}/log_transformer_model.pkl', 'wb') as file:
                pickle.dump(self.columns, file)
        return self

    def transform(self, X):
        """
        Aplica a transformação logarítmica nas colunas especificadas.
        """
        new_X = X.copy()
        for col in self.columns:
            if (new_X[col] <= 0).any():
                raise ValueError(f"A coluna {col} contém valores não positivos que não podem ser transformados.")
            
            new_X[col] = np.log1p(new_X[col])
        return new_X
    
class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Transforma colunas específicas de um DataFrame aplicando a transformação Box-Cox.
    
    Parâmetros:
    columns: list of str, opcional
        Nomes das colunas no DataFrame a serem transformadas.
    model_path: str, opcional
        Caminho para um arquivo pickle contendo um modelo de transformação Box-Cox pré-treinado.
    """

    def __init__(self, columns=None, model_path=None):
        self.columns = columns
        self.model_path = model_path
        self.transformers = {}

    def fit(self, X, y=None):
        """
        Ajusta o transformador para cada coluna especificada usando a transformação Box-Cox.
        """
        if self.model_path is not None:
            with open(self.model_path, 'rb') as file:
                saved_data = pickle.load(file)
                self.columns = saved_data['columns']
                self.transformers = saved_data['transformers']

        if self.columns is None:
            raise ValueError("Colunas não foram especificadas. Você deve especificar as colunas.")

        for col in self.columns:
            # Verifica se a coluna existe em X e se todos os valores são positivos
            if col not in X.columns:
                raise ValueError(f"A coluna '{col}' não está presente no DataFrame.")
            
            if (X[col] <= 0).any():
                raise ValueError(f"A coluna '{col}' contém valores não positivos, inadequados para a transformação Box-Cox.")

            transformer = PowerTransformer(method='box-cox', standardize=False)
            self.transformers[col] = transformer.fit(X[[col]])

        # Salvar as colunas e os transformadores em um arquivo pickle
        if self.model_path is None:
            preprocesspath = Path('../preprocessing')
            self.model_path = f'{preprocesspath}/boxcox_transformer_model.pkl'

        with open(self.model_path, 'wb') as file:
            pickle.dump({'columns': self.columns, 'transformers': self.transformers}, file)
        return self

    def transform(self, X):
        """
        Aplica a transformação Box-Cox nas colunas especificadas.
        """
        new_x = X.copy()

        for col, transformer in self.transformers.items():
            new_x[col] = transformer.transform(new_x[[col]])
        return new_x

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Um transformador que remove colunas especificadas de um DataFrame.

    Parâmetros:
    drop_columns: list, opcional
        Uma lista de nomes de colunas a serem removidas do DataFrame.

    Métodos:
    fit: Não faz nenhuma operação de ajuste, retorna o próprio objeto.
    transform: Remove as colunas especificadas do DataFrame.
    """

    def __init__(self, drop_columns=None):
        """
        Inicializa a classe com as colunas a serem removidas.
        """
        self.drop_columns = drop_columns if drop_columns is not None else []

    def fit(self, X, y=None):
        """
        Método de ajuste. Não realiza nenhuma operação.
        Retorna a própria instância da classe.
        """
        return self

    def transform(self, X):
        """
        Remove as colunas especificadas do DataFrame.
        Retorna o DataFrame com as colunas removidas.
        """
        X = X.copy()
        X.drop(self.drop_columns, axis=1, inplace=True, errors='ignore')
        return X

class ServiceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        """
        Inicializa a classe com as colunas nas quais foram parametrizadas para realizar a validação e transformação.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """
        Método de ajuste. Não realiza nenhuma operação.
        Retorna a própria instância da classe.
        """
        return self

    def transform(self, X):
        """
        Realiza a validação e transformação nas colunas especificadas.
        Se encontrar "No [alguma coisa] service" e a validação for bem-sucedida, transforma em "No".
        """
        for column in self.columns:
            if column in X.columns:
                # Aplica a transformação condicional
                X[column] = X.apply(lambda row: self._validate_and_transform(row, column), axis=1)
        return X

    def _validate_and_transform(self, row, column):
        value = row[column]

        if 'No phone service' in value:
            if row['Phone Service'] == 'No':
                return 'No'
            
        elif 'No internet service' in value:
            if row['Internet Service'] == 'No':
                return 'No'
        return value

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Um transformador que aplica OneHotEncoder ou LabelEncoder às colunas especificadas, 
    ou a todas as colunas categóricas se nenhuma coluna específica for fornecida.
    No caso do OneHotEncoder, ele também lida com a multicolinearidade removendo a 
    primeira coluna codificada se necessário.

    Parâmetros:
    encoder_type: str, default='onehot'
        O tipo de encoder a ser aplicado ('onehot' ou 'label').
    drop: str ou None, default='first'
        Se 'first', a primeira coluna de cada codificação one-hot é descartada para
        evitar multicolinearidade. Se None, todas as colunas são mantidas.
    specified_columns: list ou None, default=None
        Uma lista de colunas especificadas para codificar. Se None, todas as colunas
        categóricas serão codificadas.

    Atributos:
    encoders: dict
        Um dicionário mapeando nomes de colunas a seus respectivos encoders.

    Métodos:
    fit(X, y=None): 
        Aprende as categorias das colunas especificadas ou categóricas.
    transform(X): 
        Transforma as colunas aplicando o encoder escolhido.
    """

    def __init__(self, encoder_type='onehot', drop='first', specified_columns=None):
        self.encoder_type = encoder_type
        self.drop = drop
        self.specified_columns = specified_columns
        self.encoders = {}

    def fit(self, X, y=None):
        columns_encode = self.specified_columns if self.specified_columns is not None else X.select_dtypes(include=['object', 'category']).columns
        
        for col in columns_encode:
            if self.encoder_type == 'onehot':
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='error', drop=self.drop).fit(X[[col]])
            
            elif self.encoder_type == 'label':
                self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        new_X = X.copy()

        for col, encoder in self.encoders.items():
            if self.encoder_type == 'onehot':
                new_col = encoder.transform(X[[col]])
                new_col = pd.DataFrame(new_col, columns=encoder.get_feature_names_out([col]))
                new_X = pd.concat([new_X, new_col], axis=1)

            elif self.encoder_type == 'label':
                new_X[col] = encoder.transform(X[col])
        
        if self.encoder_type == 'onehot':
            new_X.drop(columns=self.encoders.keys(), axis=1, inplace=True)
        return new_X
    
class RBFTransformer(BaseEstimator, TransformerMixin):
    """
    Transforma uma coluna específica de um DataFrame aplicando o kernel RBF com base nas principais modas (picos).

    Parâmetros:
    column: str
        Nome da coluna no DataFrame a ser transformada.
    n_modes: int
        Número de modas (picos) a serem identificados nos dados.
    gamma: float
        Parâmetro gamma do kernel RBF que define a largura da curva de similaridade.
    model_path: str, opcional
        Caminho para um arquivo pickle contendo as modas identificadas.
    """
    
    def __init__(self, column, n_modes=2, gamma=0.1, model_path=None):
        self.column = column
        self.n_modes = n_modes
        self.gamma = gamma
        self.model_path = model_path
        self.modes_ = None

    def fit(self, X, y=None):
        """
        Ajusta o transformador identificando as modas da distribuição da coluna especificada de X.
        """
        if self.model_path is not None:
            with open(self.model_path, 'rb') as file:
                self.modes_ = pickle.load(file)
                
        if self.modes_ is None:
            column_data = X[self.column].values
            peaks, _ = find_peaks(column_data, distance=1)  # 'distance' pode ser ajustado se for necessário.
            
            if len(peaks) < self.n_modes:
                raise RuntimeError("Número de modas encontradas menor que 'n_modes'.")
            prominences = column_data[peaks]
            # Armazenar os valores das modas
            self.modes_ = column_data[peaks[np.argsort(prominences)[-self.n_modes:]]]

            if self.model_path is None:
                preprocesspath = Path('../preprocessing')
                self.model_path = f'{preprocesspath}/rbf_transformer_model.pkl'

            with open(self.model_path, 'wb') as file:
                pickle.dump(self.modes_, file)
        return self

    def transform(self, X):
        """
        Aplica a transformação RBF na coluna especificada de X usando as modas identificadas.
        """
        if self.modes_ is None:
            raise RuntimeError("Chame 'fit' antes de 'transform'.")

        column_data = X[self.column].values.reshape(-1, 1)

        # Usar os valores das modas para encontrar índices correspondentes no conjunto de dados atual
        for i, moda_valor in enumerate(self.modes_):
            # Encontrar o índice mais próximo do valor da moda no conjunto de dados atual
            moda_index = np.abs(column_data - moda_valor).argmin()
            X_rbf = rbf_kernel(column_data, column_data[moda_index].reshape(-1, 1), gamma=self.gamma)
            X[f"{self.column}_mode_{i}"] = X_rbf.flatten()
        return X
    
class KMeansCluster(BaseEstimator, TransformerMixin):
    """
    Um transformador que utiliza um modelo KMeans para agrupar dados e adicionar uma nova coluna ao DataFrame 
    com os rótulos dos clusters. Essa classe pode carregar um modelo KMeans pré-treinado de um arquivo pickle 
    ou treinar um novo modelo KMeans e salvá-lo automaticamente se nenhum caminho for especificado.

    Parâmetros:
    model_path: str, opcional
        Caminho para um arquivo pickle contendo um modelo KMeans pré-treinado. Se não fornecido, um novo modelo será treinado e salvo.
    n_clusters: int, opcional
        Número de clusters para o KMeans. Usado apenas se um novo modelo for treinado.
    columns_cluster: list
        As colunas do DataFrame a serem usadas para o clustering.

    Atributos:
    kmeans: KMeans
        Instância do modelo KMeans carregada do arquivo pickle ou treinada.
    """

    def __init__(self, model_path=None, n_clusters=3, columns_cluster=None):
        self.model_path = model_path
        self.n_clusters = n_clusters
        self.columns_cluster = columns_cluster
    
    def fit(self, X, y=None):
        """
        Método de ajuste. Treina o modelo KMeans se nenhum modelo pré-treinado foi carregado.
        Salva o modelo treinado em um arquivo pickle se um caminho for fornecido, ou cria um nome padrão para o arquivo.
        """
        if self.model_path is not None and os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as file:
                self.kmeans = pickle.load(file)
        else:
            self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++')

        if self.columns_cluster is None:
            raise ValueError("As colunas para clustering não foram especificadas.")

        # Treinar o modelo KMeans
        self.kmeans.fit(X[self.columns_cluster])
        
        # Salvar o modelo treinado
        if self.model_path is None:
            preprocesspath = Path('../preprocessing')
            with open(f'{preprocesspath}/kmeans_model.pkl', 'wb') as file:
                pickle.dump(self.kmeans, file)
        return self

    def transform(self, X):
        """
        Aplica o modelo KMeans aos dados especificados e adiciona uma coluna 'Cluster' ao DataFrame
        com os rótulos dos clusters.
        """
        if self.columns_cluster is None:
            raise ValueError("As colunas para clustering não foram especificadas.")

        X = X.copy()
        X['Cluster'] = self.kmeans.predict(X[self.columns_cluster])
        return X