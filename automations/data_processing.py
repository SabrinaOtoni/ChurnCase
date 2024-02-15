'''
Script python para tratamento do dataset "telco_customer_churn" disponibilizado pela IBM para demonstração da ferramenta IBM Cognos Analytics.
--------------------------------------
Sabrina Otoni da Silva - 2024/01
'''
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

from scipy.signal import find_peaks

import pickle

class DataTreat(BaseEstimator, TransformerMixin):
    """
    Aplica tratamentos ao DataFrame.

    Métodos:
    fit: Não faz nenhuma operação de ajuste, retorna o próprio objeto.
    transform: Aplica transformações especificadas a um DataFrame, retorna o DataFrame tratado.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, df):
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
    Aplica a transformação logarítmica em colunas de um DataFrame. 
    Essa classe pode ser inicializada com um caminho para um modelo treinado salvo como um arquivo pickle 
    ou pode ser treinada com um novo conjunto de dados.

    Parâmetros:
    columns: list of str, opcional
        Nomes das colunas no DataFrame a serem transformadas.
    model_path: str, opcional
        Caminho para um arquivo pickle contendo um modelo de transformação logarítmica pré-treinado.

    Métodos:
    fit: Método de ajuste que define as colunas para transformação.
    transform: Aplica a transformação logarítmica nas colunas especificadas.
    """
    def __init__(self, columns=None, model_path=None):
        self.columns = columns
        self.model_path = model_path

    def fit(self, X, y=None):
        if self.model_path is not None:
            with open(self.model_path, 'rb') as file:
                self.columns = pickle.load(file)

        if self.columns is None:
            raise ValueError("Colunas não foram especificadas. Você deve especificar as colunas.")
        
        if self.model_path is None:
            missing_cols = [col for col in self.columns if col not in X.columns]
            
            if missing_cols:
                raise ValueError(f"As colunas {missing_cols} não estão presentes no DataFrame.")
            
            preprocesspath = Path('../preprocessing')
            
            with open(f'{preprocesspath}/log_transformer_model.pkl', 'wb') as file:
                pickle.dump(self.columns, file)
        return self

    def transform(self, X):
        new_X = X.copy()

        for col in self.columns:
            if (new_X[col].values <= 0).any():
                raise ValueError(f"A coluna {col} contém valores não positivos que não podem ser transformados.")
            
            new_X[col] = np.log1p(new_X[col])

        if new_X.isnull().any().any():
            print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
        return new_X
    
class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Transforma colunas de um DataFrame aplicando a transformação Box-Cox.
    
    Parâmetros:
    columns: list of str, opcional
        Nomes das colunas no DataFrame a serem transformadas.
    model_path: str, opcional
        Caminho para um arquivo pickle contendo um modelo de transformação Box-Cox pré-treinado.

    Métodos:
    fit: Ajusta o transformador para cada coluna especificada usando a transformação Box-Cox.
    transform: Aplica a transformação Box-Cox nas colunas especificadas.
    """
    def __init__(self, columns=None, model_path=None):
        self.columns = columns
        self.model_path = model_path
        self.transformers = {}

    def fit(self, X, y=None):
        if self.model_path is not None:
            with open(self.model_path, 'rb') as file:
                saved_data = pickle.load(file)
                self.columns = saved_data['columns']
                self.transformers = saved_data['transformers']

        if self.columns is None:
            raise ValueError("Colunas não foram especificadas. Você deve especificar as colunas.")

        for col in self.columns:
            if col not in X.columns:
                raise ValueError(f"A coluna '{col}' não está presente no DataFrame.")
            
            if (X[col] <= 0).any():
                raise ValueError(f"A coluna '{col}' contém valores não positivos, inadequados para a transformação Box-Cox.")

            transformer = PowerTransformer(method='box-cox', standardize=False)
            self.transformers[col] = transformer.fit(X[[col]])

        if self.model_path is None:
            preprocesspath = Path('../preprocessing')
            self.model_path = f'{preprocesspath}/boxcox_transformer_model.pkl'

        with open(self.model_path, 'wb') as file:
            pickle.dump({'columns': self.columns, 'transformers': self.transformers}, file)
        return self

    def transform(self, X):
        new_x = X.copy()

        for col, transformer in self.transformers.items():
            new_x[col] = transformer.transform(new_x[[col]])

        if new_x.isnull().any().any():
            print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
        return new_x

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Remove colunas especificadas de um DataFrame.

    Parâmetros:
    drop_columns: list, opcional
        Uma lista de nomes de colunas a serem removidas do DataFrame.

    Métodos:
    fit: Não faz nenhuma operação de ajuste, retorna o próprio objeto.
    transform: Remove as colunas especificadas do DataFrame e retorna o DataFrame com as colunas removidas.
    """
    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns if drop_columns is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            X.drop(self.drop_columns, axis=1, inplace=True, errors='ignore')
        elif isinstance(X, pd.DataFrame):
            X = X.copy()
            X.drop(self.drop_columns, axis=1, inplace=True, errors='ignore')
        else:
            raise TypeError("O input precisa ser um DataFrame do pandas ou um numpy.ndarray")
        
        if X.isnull().any().any():
            print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
        return X
    
class ServiceTransformer(BaseEstimator, TransformerMixin):
    """
    Realiza a validação e transformação nas colunas especificadas. 
    Se encontrar "No [alguma coisa] service" e a validação for bem-sucedida, transforma em "No".

    Parâmetros:
    columns: list
        Uma lista de nomes de colunas a serem validadas e transformadas.

    Métodos:
    fit: Não faz nenhuma operação de ajuste, retorna o próprio objeto.
    transform: Converte X para DataFrame se necessário e realiza a validação e transformação nas colunas especificadas.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("O input precisa ser um DataFrame do pandas")

        for column in self.columns:
            if column in X.columns:
                X[column] = X.apply(lambda row: self._validate_transform(row, column), axis=1)

        if X.isnull().any().any():
            print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
        return X

    def _validate_transform(self, row, column):
        value = row[column]

        if 'No phone service' in value and row['Phone Service'] == 'No':
            return 'No'
        elif 'No internet service' in value and row['Internet Service'] == 'No':
            return 'No'
        return value
    
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Aplica get_dummies ou LabelEncoder às colunas especificadas, ou a todas as colunas categóricas se nenhuma coluna específica for fornecida.
    No caso de get_dummies, ele também lida com a multicolinearidade removendo a última coluna codificada se necessário.

    Parâmetros:
    encoder_type: str, default='onehot'
        O tipo de encoder a ser aplicado ('onehot' ou 'label').
    drop_last: bool, default=True
        Se True, a última coluna de cada codificação one-hot é descartada para evitar multicolinearidade.
    specified_columns: list ou None, default=None
        Uma lista de colunas especificadas para codificar. Se None, todas as colunas categóricas serão codificadas.
    """
    def __init__(self, encoder_type='onehot', drop=True, specified_columns=None):
        self.encoder_type = encoder_type
        self.drop = drop
        self.specified_columns = specified_columns
        self.encoders = {}
        self.train_columns = None

    def fit(self, X, y=None):
        if self.encoder_type == 'onehot':
            if self.specified_columns:
                self.train_columns = self.specified_columns
            else:
                self.train_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
        elif self.encoder_type == 'label':
            columns_encode = self.specified_columns if self.specified_columns is not None else X.select_dtypes(include=['object', 'category']).columns
            for col in columns_encode:
                self.encoders[col] = LabelEncoder().fit(X[col])

        return self

    def transform(self, X):
        new_X = X.copy()

        if self.encoder_type == 'onehot':
            dummies = pd.get_dummies(new_X[self.train_columns], drop_first=self.drop) #OneHotEncoder do Sklearn não funciona no pipeline (ValueError: Input contains NaN). Solução: personalizar o get_dummies do pandas.
            new_X = pd.concat([new_X, dummies], axis=1)
            new_X.drop(columns=self.train_columns, axis=1, inplace=True)

        elif self.encoder_type == 'label':
            for col, encoder in self.encoders.items():
                new_X[col] = encoder.transform(new_X[col])

        if new_X.isnull().any().any():
            print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
        return new_X

# class CategoricalEncoder(BaseEstimator, TransformerMixin):
#     """
#     Aplica OneHotEncoder ou LabelEncoder às colunas especificadas, ou a todas as colunas categóricas se nenhuma coluna específica for fornecida.
#     No caso do OneHotEncoder, ele também lida com a multicolinearidade removendo a primeira coluna codificada se necessário.

#     Parâmetros:
#     encoder_type: str, default='onehot'
#         O tipo de encoder a ser aplicado ('onehot' ou 'label').
#     drop: str ou None, default='first'
#         Se 'first', a primeira coluna de cada codificação one-hot é descartada para evitar multicolinearidade. Se None, todas as colunas são mantidas.
#     specified_columns: list ou None, default=None
#         Uma lista de colunas especificadas para codificar. Se None, todas as colunas categóricas serão codificadas.

#     Métodos:
#     fit: Aprende as categorias das colunas especificadas ou categóricas.
#     transform: Transforma as colunas aplicando o encoder escolhido.
#     """
#     def __init__(self, encoder_type='onehot', drop='first', specified_columns=None):
#         self.encoder_type = encoder_type
#         self.drop = drop
#         self.specified_columns = specified_columns
#         self.encoders = {}

#     def fit(self, X, y=None):
#         columns_encode = self.specified_columns if self.specified_columns is not None else X.select_dtypes(include=['object', 'category']).columns
        
#         for col in columns_encode:
#             if self.encoder_type == 'onehot':
#                 self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='error', drop=self.drop).fit(X[[col]])
            
#             elif self.encoder_type == 'label':
#                 self.encoders[col] = LabelEncoder().fit(X[col])
#         return self

#     def transform(self, X):
#         new_X = X.copy()

#         for col, encoder in self.encoders.items():
#             if self.encoder_type == 'onehot':
#                 new_col = encoder.transform(X[[col]])
#                 new_col = pd.DataFrame(new_col, columns=encoder.get_feature_names_out([col]))
#                 new_X = pd.concat([new_X, new_col], axis=1)

#             elif self.encoder_type == 'label':
#                 new_X[col] = encoder.transform(X[col])
        
#         if self.encoder_type == 'onehot':
#             new_X.drop(columns=self.encoders.keys(), axis=1, inplace=True)

#         if new_X.isnull().any().any():
#             print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
#         return new_X
    
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

    Métodos:
    fit: Ajusta o transformador identificando as modas da distribuição da coluna especificada de X.
    transform: Aplica a transformação RBF na coluna especificada de X usando as modas identificadas.
    """
    
    def __init__(self, column, n_modes=2, gamma=0.1, model_path=None, active=True):
        self.column = column
        self.n_modes = n_modes
        self.gamma = gamma
        self.model_path = model_path
        self.modes_ = None
        self.active = active

    def fit(self, X, y=None):
        if not self.active:
            return self
        else:
            if self.model_path is not None:
                with open(self.model_path, 'rb') as file:
                    self.modes_ = pickle.load(file)
                
            if self.modes_ is None:
                column_data = X[self.column].values
                peaks, _ = find_peaks(column_data, distance=1)  #'distance' pode ser ajustado.
            
                if len(peaks) < self.n_modes:
                    raise RuntimeError("Número de modas encontradas menor que 'n_modes'.")
                prominences = column_data[peaks]
                self.modes_ = column_data[peaks[np.argsort(prominences)[-self.n_modes:]]]

                if self.model_path is None:
                    preprocesspath = Path('../preprocessing')
                    self.model_path = f'{preprocesspath}/rbf_transformer_model.pkl'

                with open(self.model_path, 'wb') as file:
                    pickle.dump(self.modes_, file)
            return self

    def transform(self, X):
        if not self.active:
            return X
        else:
            if self.modes_ is None:
                raise RuntimeError("Chame 'fit' antes de 'transform'.")

            column_data = X[self.column].values.reshape(-1, 1)

            for i, moda_valor in enumerate(self.modes_):
                moda_index = np.abs(column_data - moda_valor).argmin()
                X_rbf = rbf_kernel(column_data, column_data[moda_index].reshape(-1, 1), gamma=self.gamma)
                X[f"{self.column}_mode_{i}"] = X_rbf.flatten()

            if X.isnull().any().any():
                print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
            return X
    
class KMeansCluster(BaseEstimator, TransformerMixin):
    """
    Utiliza um modelo KMeans para agrupar dados em clusters e adicionar uma nova coluna ao DataFrame com os rótulos dos clusters. 
    Essa classe pode carregar um modelo KMeans pré-treinado de um arquivo pickle ou treinar um novo modelo KMeans e salvar automaticamente se nenhum caminho de modelo for especificado.

    Parâmetros:
    model_path : str, opcional
        Caminho para um arquivo pickle contendo um modelo KMeans pré-treinado. Se não fornecido, um novo modelo será treinado e salvo.
    n_clusters : int, opcional
        Número de clusters para o KMeans. Usado apenas se um novo modelo for treinado.
    columns_cluster : list, opcional
        As colunas do DataFrame a serem usadas para o clustering.

    Métodos:
    fit: Treina o modelo KMeans com os dados fornecidos ou carrega um modelo pré-existente do caminho do arquivo especificado. Se nenhum caminho for fornecido, salva o modelo treinado em um arquivo pickle padrão.
    transform: Aplica o modelo KMeans aos dados especificados e adiciona uma nova coluna 'Cluster' ao DataFrame com os rótulos dos clusters.
    """
    def __init__(self, model_path=None, n_clusters=3, columns_cluster=None, active=True):
        self.model_path = model_path
        self.n_clusters = n_clusters
        self.columns_cluster = columns_cluster
        self.kmeans = None
        self.active = active

    def fit(self, X, y=None):
        if not self.active:
            return self
        else:
            if self.model_path is not None and os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as file:
                    self.kmeans = pickle.load(file)
            else:
                if self.columns_cluster is None:
                    raise ValueError("As colunas para clustering não foram especificadas.")
            
                self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++')
                self.kmeans.fit(X[self.columns_cluster])

                if self.model_path is None:
                    preprocesspath = Path('../preprocessing')
                    self.model_path = f'{preprocesspath}/kmeans_model.pkl'
            
                with open(self.model_path, 'wb') as file:
                    pickle.dump(self.kmeans, file)
            return self

    def transform(self, X):
        if not self.active:
            return X
        else:
            if self.kmeans is None:
                raise RuntimeError("O modelo KMeans não foi carregado ou treinado. Chame 'fit' antes de 'transform'.")
        
            if self.columns_cluster is None:
                raise ValueError("As colunas para clustering não foram especificadas.")

            X = X.copy()
            X['Cluster'] = self.kmeans.predict(X[self.columns_cluster])

            if X.isnull().any().any():
                print(f"NaNs introduzidos após a transformação {self.__class__.__name__}")
            return X