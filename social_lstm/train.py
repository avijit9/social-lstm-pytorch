'''
Train script for the Social LSTM model

Author: Anirudh Vemula
Date: 13th June 2017
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
import os
import time
import pickle
import ipdb

from model import SocialLSTM
from utils import DataLoader
from grid import getSequenceGridMask
from st_graph import ST_GRAPH
from criterion import Gaussian2DLikelihood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')
    args = parser.parse_args()
    train(args)


def train(args):
    datasets = range(5)
    # datasets = [0, 1, 3]
    datasets.remove(args.leaveDataset)
    # datasets = [0]

    dataloader = DataLoader(args.batch_size, args.seq_length+1, datasets, forcePreProcess=True)
    stgraph = ST_GRAPH(args.batch_size, args.seq_length + 1)

    # Log directory
    log_directory = 'log/'
    log_directory += str(args.leaveDataset) + '/'

    # Logging files
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = 'save/'
    save_directory += str(args.leaveDataset) + '/'

    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    def checkpoint_path(x):
        return os.path.join(save_directory, 'social_lstm_model_'+str(x)+'.tar')

    net = SocialLSTM(args)
    net.cuda()

    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)
    learning_rate = args.learning_rate

    print 'Training begin'
    best_val_loss = 100
    best_epoch = 0

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        learning_rate *= args.decay_rate
        
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0
        for batch in range(dataloader.num_batches):
            start = time.time()
            x, _, d = dataloader.next_batch()

            stgraph.readGraph(x)

            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                x_seq, d_seq = x[sequence], d[sequence]                

                if d_seq == 0 and datasets[0] == 0:
                    dataset_data = [640, 480]
                else:
                    dataset_data = [720, 576]

                grid_seq = getSequenceGridMask(x_seq, dataset_data, args.neighborhood_size, args.grid_size)

                nodes, _, nodesPresent, _ = stgraph.getSequence(sequence)

                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                numNodes = nodes.size()[1]
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()

                net.zero_grad()
                optimizer.zero_grad()

                outputs, _, _ = net(nodes[:-1], grid_seq[:-1], nodesPresent[:-1], hidden_states, cell_states)

                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:])
                
                loss_batch += loss.data[0]

                loss.backward()

                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                optimizer.step()

            stgraph.reset()
            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

        loss_epoch /= dataloader.num_batches
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')
        print '*************'
        # Validation
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch = 0

        for batch in range(dataloader.valid_num_batches):
            # Get batch data
            x, _, d = dataloader.next_valid_batch(randomUpdate=False)

            # Read the st graph from data
            stgraph.readGraph(x)

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                x_seq, d_seq = x[sequence], d[sequence]

                if d_seq == 0 and datasets[0] == 0:
                    dataset_data = [640, 480]
                else:
                    dataset_data = [720, 576]

                grid_seq = getSequenceGridMask(x_seq, dataset_data, args.neighborhood_size, args.grid_size)
                
                nodes, _, nodesPresent, _ = stgraph.getSequence(sequence)
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
    
                numNodes = nodes.size()[1]                
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()

                outputs, _, _ = net(nodes[:-1], grid_seq[:-1], nodesPresent[:-1], hidden_states, cell_states)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:])

                loss_batch += loss.data[0]

            stgraph.reset()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

        loss_epoch = loss_epoch / dataloader.valid_num_batches

        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = epoch

        print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
        print 'Best epoch', best_epoch, 'Best validation loss', best_val_loss
        log_file_curve.write(str(loss_epoch)+'\n')
        print '*****************'

        # Save the model after each epoch
        print 'Saving model'
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    print 'Best epoch', best_epoch, 'Best validation Loss', best_val_loss
    log_file.write(str(best_epoch)+','+str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()

if __name__ == '__main__':
    main()

