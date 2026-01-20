# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import argparse
import os
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils

def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--tune_from', default='',
                        help='Load the model from a .pth file')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    local_rank = setup_distributed()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)
    
    train_sampler = DistributedSampler(opencood_train_dataset)
    val_sampler = DistributedSampler(opencood_validate_dataset)
    
    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=32,
                              persistent_workers=True,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              sampler=train_sampler,
                              pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=32,
                            persistent_workers=True,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            sampler=val_sampler,
                            pin_memory=False,
                            drop_last=True)

    print('Creating Model')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:{local_rank}')

    model = DDP(train_utils.create_model(hypes).to(device), \
                device_ids=[local_rank],find_unused_parameters=True)


    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir + f'_{dist.get_rank()}'
        init_epoch, model, state_dict = \
            train_utils.load_saved_model(saved_path, model)
        
    elif opt.tune_from:
        saved_path = train_utils.setup_train(hypes,distributed=dist.get_rank())
        _, _, _ = train_utils.load_saved_model(opt.tune_from, model)
        init_epoch = 0

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes,distributed=dist.get_rank())

    # record training
    writer = SummaryWriter(saved_path)
    print(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(0, max(epoches, 0)):
        train_sampler.set_epoch(epoch)
        scheduler.step()

        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data_list in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = batch_data_list[0]

            batch_data_list = train_utils.to_device(batch_data_list, device)
            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            ouput_dict = model(batch_data_list)
            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            if dist.get_rank() == 0:
                criterion.logging(init_epoch + epoch + 1, i, len(train_loader), writer, pbar=pbar2)
                pbar2.update(1)
            # back-propagation
            final_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (init_epoch + epoch + 1)))

        if (epoch+1) % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'], train = False)
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f on GPU %d' % (init_epoch + epoch + 1,
                                                  valid_ave_loss, dist.get_rank()))
            writer.add_scalar('Validate_Loss_%d' % dist.get_rank(), valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
