# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import warnings
import sys

import yaml
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shapely\\..*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered in intersection.*", category=RuntimeWarning)

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument(
        '--init_model_dir',
        default='',
        help='Initialize weights from a checkpoint directory, but start a new run (do not resume epoch/optimizer).',
    )
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=0,
        help='Optional cap on train steps per epoch (debug only). 0 means no cap.',
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=0,
        help='Optional cap on total epochs to run (debug only). 0 means no cap.',
    )
    parser.add_argument(
        '--max_val_steps',
        type=int,
        default=0,
        help='Optional cap on validation steps per eval (debug only). 0 means no cap.',
    )
    parser.add_argument(
        '--no_test',
        action='store_true',
        help='Skip running inference.py after training finishes.',
    )
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # Resume from last checkpoint OR initialize weights from another run.
    if opt.model_dir:
        saved_path = opt.model_dir
        if opt.init_model_dir:
            _loaded_epoch, model = train_utils.load_saved_model(opt.init_model_dir, model)
            print(f"initialized model weights from {opt.init_model_dir} (checkpoint epoch {_loaded_epoch}).")
            init_epoch = 0
            scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        else:
            # Resume training from the latest epoch checkpoint (do not roll back to bestval).
            init_epoch, model = train_utils.load_latest_model(saved_path, model)
            lowest_val_epoch = init_epoch
            scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
            print(f"resume from {init_epoch} epoch.")

    else:
        # Optionally initialize model weights from another checkpoint, but start a new run.
        if opt.init_model_dir:
            _loaded_epoch, model = train_utils.load_saved_model(opt.init_model_dir, model)
            print(f"initialized model weights from {opt.init_model_dir} (checkpoint epoch {_loaded_epoch}).")
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # If a user provided an explicit --model_dir for a *new* run, config.yaml may
    # not exist yet (yaml_utils.load_yaml falls back to -y in that case). Save
    # a resolved config + scripts snapshot for reproducibility.
    os.makedirs(saved_path, exist_ok=True)
    config_path = os.path.join(saved_path, "config.yaml")
    if not os.path.exists(config_path):
        try:
            with open(config_path, "w") as f:
                yaml.dump(hypes, f)
            train_utils.backup_script(saved_path)
        except Exception as e:
            print(f"warning: failed to write config/scripts snapshot to {saved_path}: {e}")

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = int(hypes['train_params']['epoches'])
    if opt.max_epochs:
        epoches = min(epoches, int(opt.max_epochs))
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            if opt.max_train_steps and i >= opt.max_train_steps:
                break
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()

            # torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    if opt.max_val_steps and i >= opt.max_val_steps:
                        break
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = not bool(opt.no_test)
    if run_test:
        fusion_method = opt.fusion_method
        # Use the current interpreter (conda env) instead of system `python`.
        cmd = f"{sys.executable} opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
