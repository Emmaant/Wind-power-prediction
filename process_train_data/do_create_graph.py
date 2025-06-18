"""
Script to create graphs for training Graph Neural Network
"""

import argparse
from box import Box
import yaml as yaml
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from utils.graph_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create graphs for GNN")
    parser.add_argument('-config_path', type=str, help="Specify file path to config file for graphs", default='process_train_data/graph_config.yml')

    args = parser.parse_args()

    config = Box.from_yaml(filename=args.config_path, Loader=yaml.FullLoader)
    simulated = config.graph_settings.simulated

    local_feat = config.graph_settings.local_features

    df = pd.read_parquet(config.graph_settings.scada_data.input_path)
    n_global_features = len(config.graph_settings.scada_data.global_features)

    windFarmCoord = pd.read_csv(config.graph_settings.windFarmCoord_path)
    print('Read data successfully')

    x_pos, y_pos, z_pos = windFarmCoord[['X (m)','Y (m)','Z (m)']].to_numpy().T
    coordinates = np.array([x_pos, y_pos]).T

    coordinates_2d = np.column_stack([x_pos, y_pos])
    distances_2d = euclidean_distances(coordinates_2d)  # (x, y) only
    distances_z = np.abs(z_pos[:, None] - z_pos[None, :])  # Z distance only

    distances_2d = (distances_2d - np.min(distances_2d)) / (np.max(distances_2d) - np.min(distances_2d))
    distances_z = (distances_z - np.min(distances_z)) / (np.max(distances_z) - np.min(distances_z))

    attr_array = np.array([distances_2d,distances_z])

    n_local_features = 0
    if config.graph_settings.local:
        n_local_features = len(config.graph_settings.local_features)

    dir_name = '_'.join(['directed',str(config.graph_settings.directed),
                        'sim',str(config.graph_settings.simulated),
                        'local',str(config.graph_settings.local),
                        'global',str(config.graph_settings.glbal),
                        'features',str(n_local_features + n_global_features),
                        'targets',str(len(config.graph_settings.targets)),
                        'D',str(config.wake_settings.D),
                        'length',str(config.wake_settings.length),
                        'width',str(config.wake_settings.base_width),
                        'angle',str(config.wake_settings.angle)])

    print(f'''Generating graph with settings
        local: {config.graph_settings.local}, 
        global: {config.graph_settings.glbal}, 
        simulated: {config.graph_settings.simulated},
        directed: {config.graph_settings.directed}''')
    
    graph = Graph(config)
    
    graph.get_data_columns(config)
    
    df_data,df_data_norm,index = graph.preprocess_data(df,config,dir_name)

    graphs = graph.create_directed_graph_geometric(df_data,df_data_norm, coordinates, attr_array)

    graphs_train = [graphs[i] for i in index[0]]
    graphs_validation = [graphs[i] for i in index[1]]
    graphs_test = [graphs[i] for i in index[2]]

    file_path = os.path.join(config.graph_settings.output_path, dir_name)
    file_name_train = 'train.pt'
    file_name_val = 'validation.pt'
    file_name_test = 'test.pt'

    graph.save_graphs_pt(graphs_train, file_path, file_name_train)
    graph.save_graphs_pt(graphs_validation, file_path, file_name_val)
    graph.save_graphs_pt(graphs_test, file_path, file_name_test)

    full_file_path = os.path.join(file_path, 'index.pkl')

    with open(full_file_path, "wb") as f:
        pickle.dump(index, f)

    