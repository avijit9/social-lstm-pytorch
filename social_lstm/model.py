'''
Model script for the Social LSTM model

Author: Anirudh Vemula
Date: 13th June 2017
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class SocialLSTM(nn.Module):

    def __init__(self, args, infer=False):
        super(SocialLSTM, self).__init__()

        self.args = args
        self.infer = infer

        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length

        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size

        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        self.relu = nn.ReLU()

    def getSocialTensor(self, grid, hidden_states):
        numNodes = grid.size()[0]
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size).cuda())
        for node in range(numNodes):
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor

    def forward(self, nodes, grids, nodesPresent, hidden_states, cell_states):
        numNodes = nodes.size()[1]

        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size)).cuda()

        for framenum in range(self.seq_length):

            nodeIDs = nodesPresent[framenum]
            # grid = grid_seq[framenum]

            if len(nodeIDs) == 0:
                continue

            list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda())
            
            nodes_current = torch.index_select(nodes[framenum], 0, list_of_nodes)
            # grid_current = torch.index_select(grids[framenum], 0, list_of_nodes)
            grid_current = grids[framenum]
            hidden_states_current = torch.index_select(hidden_states, 0, list_of_nodes)
            cell_states_current = torch.index_select(cell_states, 0, list_of_nodes)
            
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)

            # Embed inputs
            input_embedded = self.relu(self.input_embedding_layer(nodes_current))
            tensor_embedded = self.relu(self.tensor_embedding_layer(social_tensor))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))

            outputs[framenum*numNodes + list_of_nodes.data] = self.output_layer(h_nodes)

            hidden_states[list_of_nodes.data] = h_nodes
            cell_states[list_of_nodes.data] = c_nodes

        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size).cuda())
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
