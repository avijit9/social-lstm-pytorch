'''
Test script for the Social LSTM model

Author: Anirudh Vemula
Date: 14th June 2017
'''

import os
import pickle
import os
import pickle
import argparse
import time
import ipdb

import torch
from torch.autograd import Variable

import numpy as np
from utils import DataLoader
from st_graph import ST_GRAPH
from model import SocialLSTM
from helper import getCoef, sample_gaussian_2d, compute_edges, get_mean_error, get_final_error
from criterion import Gaussian2DLikelihood, Gaussian2DLikelihoodInference
from grid import getSequenceGridMask, getGridMaskInference

def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=3,
                        help='Dataset to be tested on')

    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=49,
                        help='Epoch of model to be loaded')


    # Parse the parameters
    sample_args = parser.parse_args()

    # Save directory
    save_directory = 'save/' + str(sample_args.test_dataset) + '/'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    net = SocialLSTM(saved_args, True)
    net.cuda()

    checkpoint_path = os.path.join(save_directory, 'social_lstm_model_'+str(sample_args.epoch)+'.tar')
    # checkpoint_path = os.path.join(save_directory, 'srnn_model.tar')
    if os.path.isfile(checkpoint_path):
        print 'Loading checkpoint'
        checkpoint = torch.load(checkpoint_path)
        # model_iteration = checkpoint['iteration']
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print 'Loaded checkpoint at epoch', model_epoch


    dataset = [sample_args.test_dataset]

    dataloader = DataLoader(1, sample_args.pred_length + sample_args.obs_length, dataset, True, infer=True)

    dataloader.reset_batch_pointer()

    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, sample_args.pred_length + sample_args.obs_length)

    results = []

    # Variable to maintain total error
    total_error = 0
    final_error = 0

    for batch in range(dataloader.num_batches):
        start = time.time()

        x, _, d = dataloader.next_batch(randomUpdate=False)

        x_seq, d_seq = x[0], d[0]

        if d_seq == 0 and dataset[0] == 0:
            dimensions = [640, 480]
        else:
            dimensions = [720, 576]

        grid_seq = getSequenceGridMask(x_seq, dimensions, saved_args.neighborhood_size, saved_args.grid_size)

        stgraph.readGraph(x)

        nodes, _, nodesPresent, _ = stgraph.getSequence(0)
        nodes = Variable(torch.from_numpy(nodes).float(), volatile=True).cuda()

        obs_nodes, obs_nodesPresent, obs_grid = nodes[:sample_args.obs_length], nodesPresent[:sample_args.obs_length], grid_seq[:sample_args.obs_length]
        
        ret_nodes = sample(obs_nodes, obs_nodesPresent, obs_grid, sample_args, net, nodes, nodesPresent, grid_seq, saved_args, dimensions)

        total_error += get_mean_error(ret_nodes[sample_args.obs_length:].data, nodes[sample_args.obs_length:].data, nodesPresent[sample_args.obs_length-1], nodesPresent[sample_args.obs_length:])
        final_error += get_final_error(ret_nodes[sample_args.obs_length:].data, nodes[sample_args.obs_length:].data, nodesPresent[sample_args.obs_length-1], nodesPresent[sample_args.obs_length:])

        end = time.time()

        print 'Processed trajectory number : ', batch, 'out of', dataloader.num_batches, 'trajectories in time', end - start

        results.append((nodes.data.cpu().numpy(), ret_nodes.data.cpu().numpy(), nodesPresent, sample_args.obs_length))

        stgraph.reset()

    print 'Total mean error of the model is ', total_error / dataloader.num_batches
    print 'Total final error of the model is ', final_error / dataloader.num_batches

    print 'Saving results'
    with open(os.path.join(save_directory, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

def sample(nodes, nodesPresent, grid, args, net, true_nodes, true_nodesPresent, true_grid, saved_args, dimensions):
    numNodes = nodes.size()[1]

    hidden_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True).cuda()
    cell_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True).cuda()

    for tstep in range(args.obs_length-1):
        out_obs, hidden_states, cell_states = net(nodes[tstep].view(1, numNodes, 2), [grid[tstep]], [nodesPresent[tstep]], hidden_states, cell_states)
        loss_obs = Gaussian2DLikelihood(out_obs, nodes[tstep+1].view(1, numNodes, 2), [nodesPresent[tstep+1]])
        print loss_obs
        raw_input()

    ret_nodes = Variable(torch.zeros(args.obs_length+args.pred_length, numNodes, 2), volatile=True).cuda()
    ret_nodes[:args.obs_length, :, :] = nodes.clone()

    prev_grid = grid[-1].clone()

    for tstep in range(args.obs_length-1, args.pred_length + args.obs_length - 1):
        outputs, hidden_states, cell_states = net(ret_nodes[tstep].view(1, numNodes, 2), [prev_grid], [nodesPresent[args.obs_length-1]], hidden_states, cell_states)
        loss_pred = Gaussian2DLikelihoodInference(outputs, true_nodes[tstep+1].view(1, numNodes, 2), nodesPresent[args.obs_length-1], [true_nodesPresent[tstep+1]])
        print loss_pred
        raw_input()

        mux, muy, sx, sy, corr = getCoef(outputs)
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent[args.obs_length-1])

        ret_nodes[tstep + 1, :, 0] = next_x
        ret_nodes[tstep + 1, :, 1] = next_y

        list_of_nodes = Variable(torch.LongTensor(nodesPresent[args.obs_length-1]), volatile=True).cuda()
        current_nodes = torch.index_select(ret_nodes[tstep+1], 0, list_of_nodes)

        # Need to have a separate function at inference time
        prev_grid = getGridMaskInference(current_nodes.data.cpu().numpy(), dimensions, saved_args.neighborhood_size, saved_args.grid_size)
        prev_grid = Variable(torch.from_numpy(prev_grid).float(), volatile=True).cuda()

    return ret_nodes

if __name__ == '__main__':
    main()
