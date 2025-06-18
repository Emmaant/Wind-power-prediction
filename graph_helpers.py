import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from torch_geometric.data import Data
import joblib

from floris import FlorisModel
from flasc.utilities import floris_tools as ftools
import topography_tools as tt
from normalizing_data import *


def convert_adjacency_matrix_to_edge_index(A):
    """
    Function which converts adjacency matrix to edge index
    """
    A = torch.tensor(A.T, dtype=torch.long)
    edge_index = torch.nonzero(A)

    return edge_index.t()


def create_edge_attribute(edge_index,attr_array):
    """
    Function which takes a numpy array of attributes with the same shape as adjacency matrix and converts it
    to follow the shape of edge_index
    """
    edge_attr = np.array(
        [
        attr_array[:,edge_index[0,i],edge_index[1,i]]
        for i in range(edge_index.shape[1])
        ], dtype=float
    )
    
    return edge_attr

def create_edge_index_directed(t_coords_2d, wind_direction, wake_length, wake_base_width, wake_angle):
    """
    Function which creates edge_index for a given wind direction
    """

    turbine_wakefields = tt.get_wakefields(t_coords_2d, wind_direction, wake_length, wake_base_width, wake_angle)
    A = tt.get_wake_matrix(t_coords_2d, turbine_wakefields)

    edge_index = convert_adjacency_matrix_to_edge_index(A)
     
    return edge_index

def create_edge_index_undirected(num_turbines):
    """
    Function which creates edge_index for a fully connected matrix
    """

    A = np.ones((num_turbines,num_turbines))
    np.fill_diagonal(A,0)

    edge_index = convert_adjacency_matrix_to_edge_index(A)
     
    return edge_index

def get_split_index(y, config):
    rng = np.random.RandomState(42)  # For reproducibility
    N = len(y)
    indices = np.arange(N)

    val_size = config.data_settings.validation_ratio
    test_size = config.data_settings.test_ratio
    sample_window = config.data_settings.sample_window
    test_val_size = int(N * (val_size + test_size))

    valid_rows = ~np.isnan(y).any(axis=1)
    valid_indices = np.where(valid_rows)[0]

    if len(valid_indices) < test_val_size:
        raise ValueError(f"Not enough valid samples. Needed: {test_val_size}, Available: {len(valid_indices)}")

    collected = set()
    attempts = 0
    max_attempts = 50000  

    while len(collected) < test_val_size and attempts < max_attempts:
        start_idx = rng.choice(valid_indices)
        following_valid = valid_indices[valid_indices >= start_idx]
        window = following_valid[:sample_window]
        collected.update(window)
        attempts += 1

    collected = list(collected)
    if len(collected) < test_val_size:
        raise ValueError(f"Only {len(collected)} valid indices collected after resampling, but {test_val_size} required.")

    if test_size == 0 and val_size == 0:
        # Use all data for training
        train_indices = np.array(indices)
        val_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
    else:
        # Proceed with normal splitting
        selected_indices = collected[:test_val_size]
        val_end = int(len(selected_indices) * val_size / (val_size + test_size))

        val_indices = np.array(list(selected_indices[:val_end]))
        test_indices = np.array(list(selected_indices[val_end:]))

        train_indices = np.setdiff1d(indices, selected_indices)
        rng.shuffle(train_indices)

    return train_indices, val_indices, test_indices

class Graph:
    def __init__(self, config):
        super().__init__()
        self.local = config.graph_settings.local
        self.glbal = config.graph_settings.glbal
        self.directed = config.graph_settings.directed
        self.simulated = config.graph_settings.simulated
        self.curtailment_mode =config.graph_settings.curtailment_mode
        self.norm_type = config.data_settings.norm_type
        self.num_turbines = config.graph_settings.num_turbines
        self.num_local_features = len(config.graph_settings.local_features)
        self.data_columns = []
        self.y_columns = []
        self.data_columns_global = []
        self.data_columns_local = []
        self.mean_edge = []
        self.std_edge = []

        self.wake_length = config.wake_settings.length
        self.wake_base_width = config.wake_settings.base_width * config.wake_settings.D
        self.wake_angle = config.wake_settings.angle

    def get_data_columns(self,config):
        if self.local:
            self.data_columns_local = []
            local_features = config.graph_settings.local_features
            for i in range(self.num_turbines):
                for column in local_features:
                    self.data_columns_local.append(f"{column}{i:03d}")

        if self.glbal:
            if self.simulated:
                self.data_columns_global = config.graph_settings.simulated_data.global_features
            else:
                self.data_columns_global = config.graph_settings.scada_data.global_features

        self.data_columns = self.data_columns_local + self.data_columns_global

    def compute_relative_angle(self, edge_index, coordinates, wd):
        angles = []

        for src, dst in zip(edge_index[0], edge_index[1]):
            src, dst = src.item(), dst.item() 
            
            src_pos = coordinates[src]
            dst_pos = coordinates[dst]
    
            dx = dst_pos[0] - src_pos[0]
            dy = dst_pos[1] - src_pos[1]
    
            # Compute the bearing of the line from Node 1 to Node 2
            # Using np.arctan2(dx, dy) so 0° is north, angles increase clockwise.
            bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    
            # Example: wind "from" 45° => actual wind direction is 45 + 180 = 225
            wd_actual = (wd + 180) % 360

            angle_diff_left = np.radians((bearing - wd_actual + 360) % 360)
            angle_diff_cos = np.cos(angle_diff_left)
            angle_diff_sin = np.sin(angle_diff_left)

            if self.norm_type == 'minmax':
                angle_diff_scaled_cos = (angle_diff_cos + 1)/2
                angle_diff_scaled_sin = (angle_diff_sin + 1)/2
    
            angles.append([angle_diff_scaled_cos,angle_diff_scaled_sin])

        angles = np.array(angles)
        if angles.T.ndim == 1:
            angles = angles[:, np.newaxis]
    
        return angles

    def preprocess_data(self,df,config, dir_name):

        num_turbines = config.graph_settings.num_turbines
        output_path = config.graph_settings.output_path
        targets = config.graph_settings.targets

        if self.simulated is True:
            self.data_columns = self.data_columns + ['wd']
        else:
            self.data_columns = self.data_columns + ['wd']

        self.y_columns = [f'{x}{str(i).zfill(3)}' for i in range(num_turbines) for x in targets]

        #NOTE: Filtrerar vi inte ut detta innan?
        ###
        if self.curtailment_mode:
            self.curt_columns = [f'{x}{str(i).zfill(3)}' for i in range(num_turbines) for x in ['Curtailment mode_']]
        else:
            self.curt_columns = []
        ###
            
        data_columns = self.data_columns + self.y_columns

        #Drop nan for both X and y
        df_data = df[data_columns + self.curt_columns].copy().dropna().reset_index(drop=True)
        print('Data left', len(df_data))

        train_indices,val_indices,test_indices = get_split_index(df_data,config)

        turbine_on_cols = [col for col in self.data_columns if col.startswith("turbine_on")]
        cos_sin_cols = [col for col in self.data_columns if "cos" in col or "sin" in col]

        # Columns to normalize with DataScaler
        X_cols_to_norm = [col for col in self.data_columns if col not in turbine_on_cols + cos_sin_cols]
        y_cols_to_norm = self.y_columns  # assuming y doesn't include turbine_on or cos/sin

        # Fit scalers
        scaler_x = DataScaler(method_scaling=self.norm_type) 
        scaler_y = DataScaler(method_scaling=self.norm_type)

        scaler_x.fit(df_data.iloc[train_indices][X_cols_to_norm])
        scaler_y.fit(df_data.iloc[train_indices][y_cols_to_norm])

        df_data = df_data.loc[:, ~df_data.columns.duplicated()]

        # Apply normalization
        X_scaled = df_data[self.data_columns].copy()
        X_scaled[X_cols_to_norm] = scaler_x.transform(df_data[X_cols_to_norm])
        X_scaled[cos_sin_cols] = (df_data[cos_sin_cols] + 1) / 2  # manual normalization

        y_scaled = df_data[self.y_columns].copy()
        y_scaled[y_cols_to_norm] = scaler_y.transform(df_data[y_cols_to_norm])

        # Combine
        df_X_scaled = pd.DataFrame(X_scaled, columns=self.data_columns, index=df_data.index)
        df_y_scaled = pd.DataFrame(y_scaled, columns=self.y_columns, index=df_data.index)
        df_data_norm = pd.concat([df_X_scaled, df_y_scaled], axis=1)[data_columns]

        dir_path = os.path.join(output_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)  
        file_path = os.path.join(dir_path, 'scaler_x.pk')
        scaler_x.save_model(file_path)

        file_path = os.path.join(dir_path, 'scaler_y.pk')
        scaler_y.save_model(file_path)

        return df_data,df_data_norm,[train_indices,val_indices,test_indices]

    def create_directed_graph_geometric(self,df_data,df_data_norm, coordinates, attr_array):
        """
        Function to create directed graphs using simplified geometric wake model.

        Args:

        Returns: 
        """

        if self.curtailment_mode:
            embedding_dim = 3
            valid_values = sorted([0.0, 3.0, 18.0, 19.0, 35.0, 36.0, 40.0, 43.0, 44.0, 45.0, 50.0, 51.0, 52.0] )  
            value_to_index = {v: i for i, v in enumerate(valid_values)}
            num_categories = len(valid_values) + 1  # +1 for unknown values
            self.embedding_layer = nn.Embedding(num_categories, embedding_dim)
            curtailment_columns = [f"Curtailment mode_{i:03d}" for i in range(self.num_turbines)]

        if self.directed is False:
            edge_index = create_edge_index_undirected(self.num_turbines)
            edge_attr = create_edge_attribute(edge_index, attr_array)
            edge_attr = torch.from_numpy(edge_attr).to(dtype=torch.float32)

        graphs = []
        for i in range(len(df_data_norm)):
            data = df_data_norm.iloc[i]
            y = data[self.y_columns].copy()

            num_targets = len(self.y_columns) // self.num_turbines  # Infer number of targets
            y_nodes = torch.tensor(y.to_numpy().reshape(self.num_turbines, num_targets), dtype=torch.float32)

            #NOTE: Remove? 
            ####
            if self.curtailment_mode:
                curtailment_values = df_data.iloc[i][curtailment_columns].to_numpy()
                unknown_index = num_categories - 1  # Assign unknowns to last index
                curtailment_indices = np.array([value_to_index.get(v, unknown_index) for v in curtailment_values])
                curtailment_indices = torch.tensor(curtailment_indices, dtype=torch.long)
                curtailment_embeddings = self.embedding_layer(curtailment_indices)  # Shape: (16, embedding_dim)
            ####

            if self.local:
                columns_local = self.data_columns_local
                x_local = data[columns_local].to_numpy().reshape(self.num_turbines, -1)
                x_local = torch.from_numpy(x_local).to(dtype=torch.float32)
            else:
                x_local = None
            
            if self.glbal is True:
                x_global = data[self.data_columns_global].to_numpy() * np.ones((self.num_turbines,len(self.data_columns_global)))
                x_global = torch.from_numpy(x_global).to(dtype=torch.float32)
            else:
                x_global = None

            if x_local is not None and x_global is not None:
                x_nodes = torch.cat([x_local, x_global], dim=1)
            elif x_local is not None:
                x_nodes = x_local
            elif x_global is not None:
                x_nodes = x_global
            else:
                x_nodes = torch.empty((self.num_turbines, 0), dtype=torch.float32)

            #NOTE: Remove? 
            ####
            if self.curtailment_mode:  # Check your condition for curtailment mode
                x_nodes = torch.cat([x_nodes, curtailment_embeddings], dim=1)
            ####

            if self.directed is True:
                wd = df_data.loc[i, 'wd']
                edge_index = create_edge_index_directed(coordinates, wd, self.wake_length, self.wake_base_width, self.wake_angle)
                edge_attr = create_edge_attribute(edge_index,attr_array)
                edge_angle = self.compute_relative_angle(edge_index, coordinates, wd)
                edge_attr = torch.tensor(np.concatenate([edge_attr, edge_angle], axis=1),dtype=torch.float)

            graph_data = Data(x=x_nodes, y=y_nodes, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_attr[:, 0]) #edge weighted graph
            graph_data.validate(raise_on_error=True)
            graphs.append(graph_data)
        
        return graphs
            
    def save_graphs_pt(self,graphs, file_path, file_name):
        """
        Function for saving graphs in .pt format
        """
        if graphs is None:
            raise ValueError("Graphs are not defined.")
        
        if not file_name.endswith(".pt"):
            raise ValueError("File name must have a '.pt' extension.")
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print(f"Directory '{file_path}' created.")
        
        full_file_path = os.path.join(file_path, file_name)

        torch.save(graphs, full_file_path)

        print(f"Graphs successfully saved as '{full_file_path}'")