import os
import math
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_loaders, get_batch, TEXT
from transformer import TransformerModel


def evaluate(eval_model, data_source, criterion, bptt):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


class Trainer(object):
    def __init__(self, model, dataloaders, optimizer, scheduler, criterion, bptt, ntokens, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.bptt = bptt
        self.train_data = dataloaders.get('train', None)
        self.val_data = dataloaders.get('val', None)
        # self.test_data = dataloaders.get('test', None)
        self.ntokens = ntokens
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.train_data = self.train_data.to(self.device)
        self.val_data = self.val_data.to(self.device)

    def train_epoch(self):
        self.model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        # ntokens = len(TEXT.vocab.stoi)
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = get_batch(self.train_data, i)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.view(-1, self.ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('{:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        batch, len(self.train_data) // self.bptt, self.scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def run(self, epochs, initial_epoch=0): 
        best_val_loss = float("inf")
        best_model = None
        for epoch in range(initial_epoch + 1, epochs + 1):
            epoch_start_time = time.time()
            self.train_epoch()
            val_loss = evaluate(self.model, self.val_data, self.criterion, self.bptt)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model

            self.scheduler.step()


def argparsing():
    # parse parameters
    parser = argparse.ArgumentParser(description="Hang Le")
    parser.add_argument("--layerdrop", type=float, default=0.0,
                        help="LayerDrop rate")
    args = parser.parse_args()
    return args


def main():
    num_gpus = torch.cuda.device_count()

    # Parameters
    args = argparsing()
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.5 # the dropout value
    bptt = 35
    lr = 5.0 # learning rate
    batch_size = 20
    layerdrop = args.layerdrop

    if num_gpus < 1:
        raise Exception('No GPUs available!')
    elif num_gpus > 1:
        lr *= num_gpus
        batch_size *= num_gpus

    # Dataset
    print('Create dataloaders')
    dataloaders, info = get_loaders(name='WikiText2', batch_size=batch_size)
    ntokens = info['ntokens']

    # Model
    print('Create model')
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout, layerdrop=layerdrop)
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Create trainer')
    trainer = Trainer(model, dataloaders, optimizer, scheduler, criterion, bptt, ntokens, device)
    print('Start training')
    trainer.run(epochs=1)

if __name__ == "__main__":
    main()
    