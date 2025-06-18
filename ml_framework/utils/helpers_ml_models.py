import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# --- Data Class ---
class Data:
    def __init__(self, config):
        self.config = config
        self.simulated = config.data_settings.simulated
        self.local_features = config.data_settings.local_features
        self.global_features = config.data_settings.global_features
        self.num_turbines = config.data_settings.num_turbines
        self.targets = config.data_settings.targets

        self.data_columns = []
        self.target_columns = []
        self.data_columns_global = []
        self.data_columns_local = []

        self.x_columns = []
        self.y_columns = []
        self.x_columns_local = []
        self.x_columns_global = []

        self.scaler = None
        self.scale_cols = []

        self.split_index_path = config.data_paths_hourly.split_index if config.data_settings.hourly_data else config.data_paths.split_index

    def get_data_columns(self):
        """Initialize global and local column names for data to be used"""
        self.data_columns_local = [f"{column}{i:03d}" for i in range(self.num_turbines)
                                   for column in self.config.data_settings.local_feature_list] if self.local_features else []

        self.data_columns_global = self.config.data_settings.global_feature_list if self.global_features else []

        self.data_columns = self.data_columns_local + self.data_columns_global

    def get_column_names_training(self, idx):
        """Gets column names for turbine to be trained"""
        if self.local_features:
            self.x_columns_local = [f"{column}{idx:03d}" for column in self.config.data_settings.local_feature_list]

        if self.global_features:
            self.x_columns_global = self.config.data_settings.global_feature_list

        self.x_columns = self.x_columns_local + self.x_columns_global
        self.y_columns = [f"{target}{idx:03d}" for target in self.targets]

    def get_index(self):
        """Read in data split index from pickle file."""
        index = []
        try:
            with open(self.split_index_path, "rb") as openfile:
                while True:
                    try:
                        index.append(pickle.load(openfile))
                    except EOFError:
                        break
        except FileNotFoundError:
            print(f"File not found: {self.split_index_path}")
        return index
    
    def scale_sin_cos_columns(self,df):
        cos_sin_cols = df.filter(regex='cos|sin')
        df[cos_sin_cols.columns]  = (cos_sin_cols + 1) / 2

        return df


    def get_data(self):
        """ Method to read and get data """

        if self.config.data_settings.hourly_data:
            data_path = self.config.data_paths_hourly.simulated_data if self.simulated else self.config.data_paths_hourly.scada_data
            df = pd.read_parquet(data_path)
        else:
            data_path = self.config.data_paths.simulated_data if self.simulated else self.config.data_paths.scada_data
            df = pd.read_parquet(data_path)

        self.target_columns = [f'{x}{str(i).zfill(3)}' for i in range(self.num_turbines) for x in self.targets]
        all_columns = self.data_columns + self.target_columns

        df = df[all_columns].copy().dropna().reset_index(drop=True)
        print('Size of dataset', len(df))

        index = self.get_index()

        df = self.scale_sin_cos_columns(df)

        cos_sin_cols = df.filter(regex='cos|sin').columns.tolist()
        turbine_on_cols = df.filter(regex='turbine_on').columns.tolist()
        exclude_cols = cos_sin_cols + turbine_on_cols

        self.scale_cols = list(set(all_columns) - set(exclude_cols))

        df_tr = df.iloc[index[0][0]][all_columns]
        df_val = df.iloc[index[0][1]][all_columns]

        self.scaler = MinMaxScaler()
        df_tr[self.scale_cols] = self.scaler.fit_transform(df_tr[self.scale_cols])
        df_val[self.scale_cols]  = self.scaler.transform(df_val[self.scale_cols])
        
        if self.config.data_settings.multitask:
            self.x_columns = self.data_columns
            self.y_columns = self.target_columns

        return df_tr, df_val
