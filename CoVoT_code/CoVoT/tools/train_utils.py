# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import os
import re
from datetime import datetime

import torch
import torch.optim as optim


def load_saved_model(saved_path, model, epoch = '-1'):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    
    def strip_ddp_prefix(state_dict):
    
    # Remove the 'module.' prefix from the keys of a state_dict saved with DDP.
    
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    def add_ddp_prefix(state_dict):
    
    # Remove the 'module.' prefix from the keys of a state_dict saved with DDP.
    
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = "module." + key
            new_state_dict[new_key] = value
        return new_state_dict

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    if epoch != '-1':
        # print('resuming by loading epoch %d' % epoch)
        state_dict = torch.load(os.path.join(saved_path,
                                'net_epoch' + epoch + '.pth'),map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        state_dict = strip_ddp_prefix(state_dict)
        model.load_state_dict(state_dict)
        return epoch, model, state_dict

    if os.path.exists(os.path.join(saved_path, 'latest.pth')):
        state_dict = torch.load(
            os.path.join(saved_path, 'latest.pth'),
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        state_dict = strip_ddp_prefix(state_dict)
        model.load_state_dict(state_dict)
        return 100, model, state_dict

    initial_epoch = findLastCheckpoint(saved_path)
    state_dict = None
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        state_dict = torch.load(os.path.join(saved_path,
                                             'net_epoch%d.pth' % initial_epoch),map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        state_dict = strip_ddp_prefix(state_dict)
        # state_dict = add_ddp_prefix(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("缺失参数 (模型需要但权重文件没有):", missing_keys)
        print("冗余参数 (权重文件有但模型不需要):", unexpected_keys)
    return initial_epoch, model, state_dict


def setup_train(hypes, distributed=False):
    """
    Create folder for saved model based on current timestamp and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if distributed is not None:
        full_path = full_path + f'_{distributed}'

    # Check if the directory already exists and append a unique identifier if it does
    if os.path.exists(full_path):
        unique_id = 1
        while os.path.exists(full_path + f"_{unique_id}"):
            unique_id += 1
        full_path = full_path + f"_{unique_id}"

    os.makedirs(full_path, exist_ok=True)
    # save the yaml file
    save_name = os.path.join(full_path, 'config.yaml')
    with open(save_name, 'w') as outfile:
        yaml.dump(hypes, outfile)

    return full_path

def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or (inputs is None):
            return inputs
        return inputs.to(device)
