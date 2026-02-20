import argparse
import os
import statistics
import glob
import warnings
import yaml
import sys
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from opencood.utils import model_utils
from icecream import ic
import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shapely\\..*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered in intersection.*", category=RuntimeWarning)

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

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
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader workers per process.',
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=0,
        help='Optional cap on total epochs to run (debug only). 0 means no cap.',
    )
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=0,
        help='Optional cap on train steps per epoch (debug only). 0 means no cap.',
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
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=int(opt.num_workers),
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=int(opt.num_workers),
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=int(opt.num_workers),
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=int(opt.num_workers),
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optional freezing logic for fine-tuning. This is useful when swapping
    # positional encodings (e.g., ProPE/VRoPE) to adapt the fusion transformer
    # without destabilizing the whole perception stack.
    train_cfg = hypes.get("train_params", {}) if isinstance(hypes, dict) else {}
    trainable_modules = train_cfg.get("trainable_modules", None)
    freeze_modules = train_cfg.get("freeze_modules", None)
    freeze_bn = bool(train_cfg.get("freeze_bn", False))

    def _matches_prefix(name: str, prefixes) -> bool:
        return any(name == p or name.startswith(p + ".") for p in prefixes)

    if trainable_modules:
        # Freeze everything, then unfreeze the requested module prefixes.
        print(f"[train_ddp] trainable_modules={trainable_modules} (freezing all others)")
        for n, p in model.named_parameters():
            p.requires_grad = _matches_prefix(n, trainable_modules)
    elif freeze_modules:
        print(f"[train_ddp] freeze_modules={freeze_modules}")
        for n, p in model.named_parameters():
            if _matches_prefix(n, freeze_modules):
                p.requires_grad = False

    if freeze_bn:
        print("[train_ddp] freeze_bn=True (setting BatchNorm layers to eval mode)")
        model.apply(model_utils.fix_bn)

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # Resume from last checkpoint OR initialize weights from another run.
    if opt.model_dir:
        saved_path = opt.model_dir
        if opt.init_model_dir:
            _loaded_epoch, model = train_utils.load_saved_model(opt.init_model_dir, model)
            print(f"initialized model weights from {opt.init_model_dir} (checkpoint epoch {_loaded_epoch}).")
            init_epoch = 0
        else:
            # Resume training from the latest epoch checkpoint (do not roll back to bestval).
            init_epoch, model = train_utils.load_latest_model(saved_path, model)
            lowest_val_epoch = init_epoch
    else:
        # Optionally initialize model weights from another checkpoint, but start a new run.
        if opt.init_model_dir:
            _loaded_epoch, model = train_utils.load_saved_model(opt.init_model_dir, model)
            print(f"initialized model weights from {opt.init_model_dir} (checkpoint epoch {_loaded_epoch}).")
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # If a user provided an explicit --model_dir for a *new* run, config.yaml may
    # not exist yet (yaml_utils.load_yaml falls back to -y in that case). Save
    # a resolved config + scripts snapshot for reproducibility.
    if opt.rank == 0:
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
        
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True) # True
        model_without_ddp = model.module


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = int(hypes['train_params']['epoches'])
    if opt.max_epochs:
        epoches = min(epoches, int(opt.max_epochs))
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        # the model will be evaluation mode during validation
        model.train()
        # model.train() toggles BN layers back to train mode, so re-freeze them
        # for fine-tuning runs that request it.
        if freeze_bn:
            model.apply(model_utils.fix_bn)
        try: # heter_model stage2
            model_without_ddp.model_train_init()
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
            batch_data = train_utils.maybe_apply_pose_provider(batch_data, hypes)
            batch_data['ego']['epoch'] = epoch
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()


        # torch.cuda.empty_cache() # it will destroy memory buffer
        #
        # Save checkpoints periodically and always save the final epoch. The original
        # OpenCOOD logic saves on `epoch % save_freq == 0` with filenames `net_epoch{epoch+1}.pth`,
        # which can miss the last epoch when `epoches` is an even number and `save_freq=2`.
        save_freq = int(hypes['train_params'].get('save_freq', 1) or 1)
        if (epoch % save_freq == 0) or (epoch == int(hypes['train_params']['epoches']) - 1):
            torch.save(model_without_ddp.state_dict(),
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
                    batch_data = train_utils.maybe_apply_pose_provider(batch_data, hypes)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    if opt.rank == 0:
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    if opt.rank == 0:
        run_test = not bool(opt.no_test)
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
        if len(bestval_model_list) > 1:
            import numpy as np
            bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
            ascending_idx = np.argsort(bestval_model_epoch_list)
            for idx in ascending_idx:
                if idx != (len(bestval_model_list) - 1):
                    os.remove(bestval_model_list[idx])

        if run_test:
            fusion_method = opt.fusion_method
            # Use the same interpreter used to launch training. This avoids
            # accidentally calling system `python` (e.g., Python 2) inside Slurm jobs.
            cmd = f"{sys.executable} opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
            print(f"Running command: {cmd}")
            os.system(cmd)


if __name__ == '__main__':
    main()
