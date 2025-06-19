'''
Script for the architecture of the gnn
'''
import torch.nn as nn
import torch 
from torch_geometric.nn import GATConv, GENConv

class MLP(nn.Module):
    def __init__(self, in_dim, layer_dim_list, norm_type):
        super().__init__()

        activation = nn.ReLU
        dim_list = [in_dim] + layer_dim_list
        fc_layers = []
        for i in range(len(dim_list) - 2):
            fc_layers += [nn.Linear(dim_list[i], dim_list[i + 1]), activation()]

        # add the output layer without activation
        fc_layers += [nn.Linear(dim_list[-2], dim_list[-1])]

        # get the normalization type to add to the output of the MLP

        if norm_type is not None:
            norm_layer = getattr(nn, norm_type)
            fc_layers.append(norm_layer(dim_list[-1]))

        # init the fully connected layers
        self.__fcs = nn.Sequential(*fc_layers)

    def forward(self, x):
        return self.__fcs(x)
 
class Encoder(nn.Module):
    def __init__(self, node_enc_mlp_layers, edge_enc_mlp_layers, node_latent_dim, edge_latent_dim, feature_dim, edge_attr_dim):
        super().__init__()
        norm_type = 'LayerNorm'
        self.node_encoder = MLP(feature_dim,node_enc_mlp_layers + [node_latent_dim], norm_type=norm_type)
        self.edge_encoder = MLP(edge_attr_dim,edge_enc_mlp_layers + [edge_latent_dim], norm_type=norm_type)

    def forward(self, edge_attr, features, batch):
        x_enc = self.node_encoder(features) 
        edge_attr_enc = self.edge_encoder(edge_attr)
        return x_enc, edge_attr_enc
    
class Decoder(nn.Module):
    def __init__(self, node_latent_dim, node_dec_mlp_layers, output_dim):
        super().__init__()
        self.node_decoder = MLP(node_latent_dim,node_dec_mlp_layers + [output_dim], norm_type=None)
    def forward(self, x):
        x = self.node_decoder(x)
        return x
    
class Processor(nn.Module):    
    def __init__(self,conv_type, num_layers,node_latent_dim, edge_latent_dim, gen_agg_type):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ProcessorBlock(conv_type, node_latent_dim, edge_latent_dim, gen_agg_type))

    def forward(self, x, edge_index, edge_attr):

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        return x

class ProcessorBlock(nn.Module):    
    def __init__(self, conv_type, node_latent_dim, edge_latent_dim, gen_agg_type):
        super().__init__()

        self.conv_type = conv_type
        if conv_type == 'GATConv':
            self.conv = GATConv(in_channels=node_latent_dim, out_channels=node_latent_dim, edge_dim=edge_latent_dim)
            self.norm = nn.LayerNorm(node_latent_dim, elementwise_affine=True)
        elif conv_type == 'GENConv':

            self.conv = GENConv(in_channels=node_latent_dim, out_channels=node_latent_dim, norm='layer',aggr=gen_agg_type, num_layers=2)
            self.norm = nn.Identity()

    def forward(self, x, edge_index, edge_attr):        
        return self.norm(self.conv(x, edge_index, edge_attr))


class WindFarmGNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        node_latent_dim = config.encoder_settings.node_latent_dim
        edge_latent_dim = config.encoder_settings.edge_latent_dim

        self.encoder = Encoder(config.encoder_settings.node_enc_mlp_layers, config.encoder_settings.edge_enc_mlp_layers,
                                node_latent_dim, edge_latent_dim, config.feature_dim, config.edge_attr_dim)
        self.processor = Processor(config.model_settings.conv_type,config.model_settings.num_layers,node_latent_dim,
                                    edge_latent_dim, config.model_settings.GEN_agg_type)
        self.decoder = Decoder(node_latent_dim, config.decoder_settings.node_dec_mlp_layers, config.decoder_settings.output_dim)

    def forward(self, data):

        x, edge_attr = self.encoder(data.edge_attr, data.x, data.batch)
        x = self.processor(x, data.edge_index, edge_attr)
        x = self.decoder(x)
        
        return x
    def compute_loss(self,y,out):
        mask_first_column = (y[:, 0] != 0) 

        if mask_first_column.sum() == 0:
            return torch.tensor(1e-12, requires_grad=True)  
        y_masked = y[mask_first_column]
        out_masked = out[mask_first_column]
        return nn.functional.mse_loss(y_masked, out_masked, reduction='mean')