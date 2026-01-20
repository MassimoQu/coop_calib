import torch
import torch.nn as nn
import torch.nn.functional as F
# from ipdb import set_trace
import random
import numpy as np
from ...utils.spconv_utils import spconv
from .norm import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d
import torch_scatter
import traceback

def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret

@torch.no_grad()
def get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    '''
    Args:
        batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to. Window inds is unique is the whole batch.
        voxel_drop_lvl: shape=[N, ]. Indicates batching_level of the window the voxel belongs to.
    Returns:
        flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
            Determine the voxel position in range [0, num_windows * max_tokens) of each voxel.
    '''
    device = batch_win_inds.device

    flat2window_inds_dict = {}

    for dl in drop_info: # dl: short for drop level

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        conti_win_inds = make_continuous_inds(batch_win_inds[dl_mask])

        max_tokens = drop_info[dl]['max_tokens']

        inner_win_inds = get_inner_win_inds(conti_win_inds)

        flat2window_inds = conti_win_inds * max_tokens + inner_win_inds

        flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

        if debug:
            num_windows = len(torch.unique(conti_win_inds))
            assert inner_win_inds.max() < max_tokens, f'Max inner inds({inner_win_inds.max()}) larger(equal) than {max_tokens}'
            assert (flat2window_inds >= 0).all()
            max_ind = flat2window_inds.max().item()
            assert  max_ind < num_windows * max_tokens, f'max_ind({max_ind}) larger than upper bound({num_windows * max_tokens})'
            assert  max_ind >= (num_windows-1) * max_tokens, f'max_ind({max_ind}) less than lower bound({(num_windows-1) * max_tokens})'

    return flat2window_inds_dict


def flat2window(feat, voxel_drop_lvl, flat2win_inds_dict, drop_info, padding=0):
    '''
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_drop_lvl: shape=[N, ]. Indicates drop_level of the window the voxel belongs to.
    Returns:
        feat_3d_dict: contains feat_3d of each drop level. Shape of feat_3d is [num_windows, num_max_tokens, C].
    
    drop_info:
    {1:{'max_tokens':50, 'range':(0, 50)}, }
    '''
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]

    feat_3d_dict = {}

    for dl in drop_info:

        dl_mask = voxel_drop_lvl == dl
        if not dl_mask.any():
            continue

        feat_this_dl = feat[dl_mask]

        this_inds = flat2win_inds_dict[dl][0]

        max_tokens = drop_info[dl]['max_tokens']
        num_windows = (this_inds // max_tokens).max().item() + 1
        padding = torch.tensor(padding, dtype=dtype, device=device)
        feat_3d = torch.ones((num_windows * max_tokens, feat_dim), dtype=dtype, device=device) * padding
        # if this_inds.max() >= num_windows * max_tokens:
        #     set_trace()
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[dl] = feat_3d

    return feat_3d_dict

def window2flat(feat_3d_dict, inds_dict):
    flat_feat_list = []

    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]
    
    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    # check_feat = -torch.ones((num_all_voxel,), device=device, dtype=torch.long)

    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat
        # check_feat[flat_pos] = 0
        # flat_feat_list.append(flat_feat)
    # assert (check_feat == 0).all()
    
    return all_flat_feat

def get_flat2win_inds_v2(batch_win_inds, voxel_drop_lvl, drop_info, debug=True):
    transform_dict = get_flat2win_inds(batch_win_inds, voxel_drop_lvl, drop_info, debug)
    # add voxel_drop_lvl and batching_info into transform_dict for better wrapping
    transform_dict['voxel_drop_level'] = voxel_drop_lvl
    transform_dict['batching_info'] = drop_info
    return transform_dict
    
def window2flat_v2(feat_3d_dict, inds_dict):
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)

def flat2window_v2(feat, inds_dict, padding=0):
    assert 'voxel_drop_level' in inds_dict, 'voxel_drop_level should be in inds_dict in v2 function'
    inds_v1 = {k:inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_drop_level'], inds_v1, batching_info, padding=padding)

def scatter_v2(feat, coors, mode, return_inv=True, min_points=0, unq_inv=None, new_coors=None):
    assert feat.size(0) == coors.size(0)
    if mode == 'avg':
        mode = 'mean'


    if unq_inv is None and min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    elif unq_inv is None:
        new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
    else:
        assert new_coors is not None, 'please pass new_coors for interface consistency, caller: {}'.format(traceback.extract_stack()[-2][2])


    if min_points > 0:
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
        feat = feat[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

    if mode == 'max':
        new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
    elif mode in ('mean', 'sum'):
        new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
    else:
        raise NotImplementedError

    if not return_inv:
        return new_feat, new_coors
    else:
        return new_feat, new_coors, unq_inv

def filter_almost_empty(pts_coors, min_points=5):
    if min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(pts_coors, return_inverse=True, return_counts=True, dim=0)
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
    else:
        valid_mask = torch.ones(len(pts_coors), device=pts_coors.device, dtype=torch.bool)
    return valid_mask


@torch.no_grad()
def get_inner_win_inds_deprecated(win_inds):
    '''
    Args:
        win_inds indicates which windows a voxel belongs to. Voxels share a window have same inds.
        shape = [N,]
    Return:
        inner_inds: shape=[N,]. Indicates voxel's id in a window. if M voxels share a window, their inner_inds would
            be torch.arange(m, dtype=torch.long)
    Note that this function might output different results from get_inner_win_inds_slow due to the unstable pytorch sort.
    '''

    sort_inds, order = win_inds.sort() #sort_inds is like [0,0,0, 1, 2,2] -> [0,1, 2, 0, 0, 1]
    roll_inds_left = torch.roll(sort_inds, -1) # [0,0, 1, 2,2,0]

    diff = sort_inds - roll_inds_left #[0, 0, -1, -1, 0, 2]
    end_pos_mask = diff != 0

    bincount = torch.bincount(win_inds)
    # assert bincount.max() <= max_tokens
    unique_sort_inds, _ = torch.sort(torch.unique(win_inds))
    num_tokens_each_win = bincount[unique_sort_inds] #[3, 1, 2]

    template = torch.ones_like(win_inds) #[1,1,1, 1, 1,1]
    template[end_pos_mask] = (num_tokens_each_win-1) * -1 #[1,1,-2, 0, 1,-1]

    inner_inds = torch.cumsum(template, 0) #[1,2,0, 0, 1,0]
    inner_inds[end_pos_mask] = num_tokens_each_win #[1,2,3, 1, 1,2]
    inner_inds -= 1 #[0,1,2, 0, 0,1]


    #recover the order
    inner_inds_reorder = -torch.ones_like(win_inds)
    inner_inds_reorder[order] = inner_inds

    ##sanity check
    assert (inner_inds >= 0).all()
    assert (inner_inds == 0).sum() == len(unique_sort_inds)
    assert (num_tokens_each_win > 0).all()
    random_win = unique_sort_inds[random.randint(0, len(unique_sort_inds)-1)]
    random_mask = win_inds == random_win
    num_voxel_this_win = bincount[random_win].item()
    random_inner_inds = inner_inds_reorder[random_mask] 

    assert len(torch.unique(random_inner_inds)) == num_voxel_this_win
    assert random_inner_inds.max() == num_voxel_this_win - 1
    assert random_inner_inds.min() == 0

    return inner_inds_reorder

from torch.autograd import Function
try:
    import ingroup_indices
except Exception:
    ingroup_indices = None

if ingroup_indices is not None:
    class IngroupIndicesFunction(Function):

        @staticmethod
        def forward(ctx, group_inds):
            out_inds = torch.zeros_like(group_inds) - 1
            ingroup_indices.forward(group_inds, out_inds)
            ctx.mark_non_differentiable(out_inds)
            return out_inds

        @staticmethod
        def backward(ctx, g):
            return None

    get_inner_win_inds = IngroupIndicesFunction.apply
else:
    def get_inner_win_inds(group_inds):
        # Fallback to the pure PyTorch implementation when the custom op is unavailable.
        return get_inner_win_inds_deprecated(group_inds)

@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift):

    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, 'Usually holds... in case of wrong order'

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1) # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1) # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1) # plus one here to meet the needs of shift.
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    if do_shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z
    
    # compatibility between 2D window and 3D window
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    # shifted_coors_x = coors[:, 3] + shift_x
    # shifted_coors_y = coors[:, 2] + shift_y
    # shifted_coors_z = coors[:, 1] + shift_z
    
    # for 2D situations
    shifted_coors_x = coors[:, 2] + shift_x
    shifted_coors_y = coors[:, 1] + shift_y
    shifted_coors_z = coors[:, 1] - coors[:, 1] + shift_z

    win_coors_x = shifted_coors_x // win_shape_x
    win_coors_y = shifted_coors_y // win_shape_y
    win_coors_z = shifted_coors_z // win_shape_z

    if len(window_shape) == 2:
        assert (win_coors_z == 0).all()

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                        win_coors_x * max_num_win_y * max_num_win_z + \
                        win_coors_y * max_num_win_z + \
                        win_coors_z

    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)
    # coors_in_win = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)
    
    return batch_win_inds, coors_in_win

@torch.no_grad()
def make_continuous_inds(inds):

    ### make batch_win_inds continuous
    dtype = inds.dtype
    device = inds.device

    unique_inds, _ = torch.sort(torch.unique(inds))
    num_valid_inds = len(unique_inds)
    max_origin_inds = unique_inds.max().item()
    canvas = -torch.ones((max_origin_inds+1,), dtype=dtype, device=device)
    canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)

    conti_inds = canvas[inds]

    return conti_inds


def build_mlp(in_channel, hidden_dims, norm_cfg, is_head=False, act='relu', bias=False, dropout=0):
    layer_list = []
    last_channel = in_channel
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims, ]
    for i, c in enumerate(hidden_dims):
        act_layer = get_activation_layer(act, c)

        norm_layer = build_norm_layer(norm_cfg, c)[1]
        if i == len(hidden_dims) - 1 and is_head:
            layer_list.append(nn.Linear(last_channel, c, bias=True),)
        else:
            sq = [
                nn.Linear(last_channel, c, bias=bias),
                norm_layer,
                act_layer,
            ]
            if dropout > 0:
                sq.append(nn.Dropout(dropout))
            layer_list.append(
                nn.Sequential(
                    *sq
                )
            )

        last_channel = c
    mlp = nn.Sequential(*layer_list)
    return mlp

def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def get_activation_layer(act, dim=None):
    """Return an activation function given a string"""
    act = act.lower()
    if act == 'relu':
        act_layer = nn.ReLU(inplace=True)
    elif act == 'gelu':
        act_layer = nn.GELU()
    elif act == 'leakyrelu':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        act_layer = nn.PReLU(num_parameters=dim)
    elif act == 'swish' or act == 'silu':
        act_layer = nn.SiLU(inplace=True)
    elif act == 'glu':
        act_layer = nn.GLU()
    elif act == 'elu':
        act_layer = nn.ELU(inplace=True)
    else:
        raise NotImplementedError
    return act_layer

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
    "naiveSyncBN1d": ("naiveSyncBN1d",NaiveSyncBatchNorm1d),
    "naiveSyncBN2d": ("naiveSyncBN2d",NaiveSyncBatchNorm2d),
}


def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    layer.apply(bn_weight_init)

    return name, layer

def bn_weight_init(m):
    if m.weight is not None:
        torch.nn.init.uniform_(m.weight)

def conv_ws_2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-5
):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

class ConvWS2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
    ):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.eps,
        )
    
conv_cfg = {
    "Conv": nn.Conv2d,
    "ConvWS": ConvWS2d,
    # TODO: octave conv
}

def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer
    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.
    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type="Conv")
    else:
        assert isinstance(cfg, dict) and "type" in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in conv_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer
