
from functools import partial
import torch
import torch.nn as nn
import copy
from opencood.models.sub_modules.sst_input_layer_v2 import SSTInputLayerV2
from opencood.models.sub_modules.sst_basic_block_v2 import BasicShiftBlockV2
from opencood.models.sub_modules.sst_ops import flat2window_v2, window2flat_v2, build_norm_layer, build_conv_layer
from ...utils.spconv_utils import replace_feature, spconv
from spconv.core import ConvAlgo
from opencood.models.sub_modules.voxelnext import SparseBasicBlock2D
from opencood.models.fuse_modules.sparse_vote_fusion import SparseVoteFusion

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CoVoT(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[256,256,256],
        nhead=[8,8,8],
        num_blocks=3,
        dim_feedforward=[256,256,256],
        vote_channel=[256,128,64],
        dropout=0.0,
        activation="gelu",
        args = None,
        output_shape=None,
        num_attached_conv=2,
        conv_in_channel=64,
        conv_out_channel=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='', bias=False),
        debug=True,
        in_channel=None,
        to_bev=True,
        conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
        checkpoint_blocks=[],
        # layer_cfg=dict(use_bn=False, cosine=True, tau_min=0.01),
        layer_cfg=dict(),
        conv_shortcut=False,
        ):
        super().__init__()
        
        self.d_model = d_model = args["d_model"]
        self.nhead = nhead = args["nhead"]
        num_blocks = args["num_blocks"]
        dim_feedforward = args["dim_feedforward"]
        vote_channel = args["vote_channel"]
        self.pruning_ratio = args["pruning_ratio"] if "pruning_ratio" in args else 0.0
        self.checkpoint_blocks = checkpoint_blocks
        self.conv_shortcut = conv_shortcut
        self.to_bev = to_bev
        
        self.ast_layer_encoder = SSTInputLayerV2(args["encoder"],mute=True)

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_voter_encoder = spconv.SparseSequential(
        spconv.SparseConv2d(d_model[-1], vote_channel[0], 3, stride=2, padding=1, bias=False, indice_key='voter_encode0', algo=ConvAlgo.Native),
        nn.BatchNorm1d(vote_channel[0]),
        nn.ReLU(True),
        SparseBasicBlock2D(vote_channel[0],vote_channel[0],norm_fn=norm_fn, indice_key='res_enc0'),
        spconv.SparseConv2d(vote_channel[0], vote_channel[1], 3, stride=2, padding=1, bias=False,indice_key='voter_encode1', algo=ConvAlgo.Native),
        nn.BatchNorm1d(vote_channel[1]),
        nn.ReLU(True),
        SparseBasicBlock2D(vote_channel[1],vote_channel[1],norm_fn=norm_fn, indice_key='res_enc1'),
        spconv.SparseConv2d(vote_channel[1], vote_channel[2], 3, stride=2, padding=1, bias=False,indice_key='voter_encode2', algo=ConvAlgo.Native),
        nn.BatchNorm1d(vote_channel[2]),
        nn.ReLU(True),
        SparseBasicBlock2D(vote_channel[2],vote_channel[2],norm_fn=norm_fn, indice_key='res_enc2'),
        )

        self.win_prob = nn.Sequential(nn.Linear(vote_channel[-1], 128),nn.ReLU(True),nn.Linear(128, 1),nn.BatchNorm1d(1),nn.Sigmoid())
        
        self.sparse_voter_decoder_with_vic = spconv.SparseSequential(
        spconv.SparseConvTranspose2d(vote_channel[2], vote_channel[1], 3 ,stride = 2, padding = 1, bias=False, indice_key='voter_decode0', algo=ConvAlgo.Native),
        nn.BatchNorm1d(vote_channel[1]),
        nn.ReLU(True),
        SparseBasicBlock2D(vote_channel[1],vote_channel[1],norm_fn=norm_fn, indice_key='res_dec0'),
        spconv.SparseConvTranspose2d(vote_channel[1], vote_channel[0], 3 , stride = 2, padding = 1, bias=False,indice_key='voter_decode1', algo=ConvAlgo.Native),
        nn.BatchNorm1d(vote_channel[0]),
        nn.ReLU(True),
        SparseBasicBlock2D(vote_channel[0],vote_channel[0],norm_fn=norm_fn, indice_key='res_dec1'),
        spconv.SparseConvTranspose2d(vote_channel[0], d_model[-1], 3 , stride = 2, padding = 1, bias=False, indice_key='voter_decode2', algo=ConvAlgo.Native),
        nn.BatchNorm1d(d_model[-1]),
        nn.ReLU(True),
        SparseBasicBlock2D(d_model[-1],d_model[-1],norm_fn=norm_fn, indice_key='res_dec2'),
        )

        # Sparse Regional Cross Attention Fusion Blocks
        block_list_cross=[]
        for i in range(num_blocks):
            block_list_cross.insert(0,
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )

        self.block_list = nn.ModuleList(block_list)
        self.block_list_cross = nn.ModuleList(block_list_cross)

        self.res_fusion = SparseVoteFusion(args['fusion_channel'])
            
        self._reset_parameters()

        self.output_shape = args["output_shape"]

        self.debug = args["debug"]

    def forward(self, batch_dict):
        '''
        '''
        ### sparse voting tranformer encoder -- self attention
        record_len = batch_dict["record_len"]
        batch_size_fused = record_len.shape[0]
        voxel_info = self.ast_layer_encoder(batch_dict)
        num_shifts = 2 
        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

        device = voxel_info['voxel_coors'].device
        batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        voxel_feat = voxel_info['voxel_feats']
        ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]

        input = voxel_feat
        if hasattr(self, 'linear0'):
            input = self.linear0(input)
        for i, block in enumerate(self.block_list):
            input = block(input, pos_embed_list, ind_dict_list, 
                padding_mask_list, using_checkpoint = False)
        ### sparse voter encoding & decoding (for communication)
        x_self_attn = spconv.SparseConvTensor(
            features=input,
            indices=voxel_info['voxel_coors'].to(torch.int32),
            spatial_shape=self.output_shape,
            batch_size=batch_size
        )

        regrouped_batch_dict = self.voting_regroup(x_self_attn, record_len)
        vote_loss = regrouped_batch_dict['vote_loss']
        
        ### sparse voting tranformer decoder -- self & cross attention
        voxel_info_col = self.ast_layer_encoder(regrouped_batch_dict)
        assert voxel_info_col['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

        device = voxel_info_col['voxel_coors'].device
        batch_size_col = voxel_info_col['voxel_coors'][:, 0].max().item() + 1
        assert batch_size_col == batch_size_fused
        voxel_feat_col = voxel_info_col['voxel_feats']
        ind_dict_list_col = [voxel_info_col[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list_col = [voxel_info_col[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list_col = [voxel_info_col[f'pos_dict_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat_col
        for i, block in enumerate(self.block_list_cross):
            output = block(output, pos_embed_list_col, ind_dict_list_col, 
                padding_mask_list_col, using_checkpoint = False)
        ### output to dense bev for anchor head
        assert self.to_bev
        if self.to_bev:       
            out_spconv_tensor = spconv.SparseConvTensor(
                features=output,
                indices=voxel_info_col['voxel_coors'].to(torch.int32),
                spatial_shape=self.output_shape,
                batch_size=batch_size_col
            )
            output = self.res_fusion(voxel_info["ego_spconv_tensor"],out_spconv_tensor)
            output = self.recover_bev(output.features, output.indices.long(), batch_size_fused)

        if not self.to_bev:
            output = {'voxel_feats':output, 'voxel_coors':voxel_info_col['voxel_coors']}

        # if len(output_list) == 0: # weird code, just for developing
        #     output_list.append(output)

        batch_dict.update({"spatial_features": output})
        batch_dict.update({"vote_loss":vote_loss})
        batch_dict.update({"comm_points":regrouped_batch_dict["comm_points"]})
        return batch_dict
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name and 'tau' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]

            # debugging
            if (coors[:, 1] > ny - 1).any() or (coors[:, 1] < 0).any() \
                or (coors[:, 2] > nx - 1).any() or (coors[:, 2] < 0).any():
                
                # find illigel cooords only for debuggiong
                invalid_y_mask = (coors[:, 1] > ny - 1) | (coors[:, 1] < 0)
                invalid_x_mask = (coors[:, 2] > nx - 1) | (coors[:, 2] < 0)
                invalid_coors = coors[invalid_y_mask | invalid_x_mask]
                
                # raise illegel!
                raise ValueError(
                    f"overflow coords\n"
                    f"shape should be (ny, nx) = {ny}, {nx}\n"
                    f"illegel coords:\n{invalid_coors[:5].cpu().numpy()}"
                )
            indices = this_coors[:, 1] * nx + this_coors[:, 2]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas

    def voting_regroup(self, x_input, record_len):
        self_indices = x_input.indices
        self_features = x_input.features
        regrouped_batch_dict = {"record_len":record_len}
        x_voter = self.sparse_voter_encoder(x_input)
        x_kept, vote_loss, comm_points = self.voting_info_confirm(x_voter,record_len)
        x_output = self.sparse_voter_decoder_with_vic(x_kept)
        x_features = x_output.features
        x_indices = x_output.indices
        # vote pruning with focal loss guidance
        cum_sum_len = torch.cumsum(record_len, dim=0)
        cum_sum_len = torch.cat([torch.tensor([0]).to(cum_sum_len.device), cum_sum_len], dim=0)
        new_feature = []
        new_indices = []
        for ego_batch_id in range(record_len.shape[0]):
                new_feature.append(self_features[self_indices[:,0]==cum_sum_len[ego_batch_id]])
                new_indices.append(self_indices[self_indices[:,0]==cum_sum_len[ego_batch_id]])
                new_indices[-1][:,0] = ego_batch_id
                new_feature.append(x_features[(x_indices[:, 0] >= (cum_sum_len[ego_batch_id])) \
                                & (x_indices[:, 0] < cum_sum_len[ego_batch_id+1])])
                new_indices.append(x_indices[(x_indices[:, 0] >= (cum_sum_len[ego_batch_id])) \
                                & (x_indices[:, 0] < cum_sum_len[ego_batch_id+1])])
                new_indices[-1][:,0] = ego_batch_id
        x_new_feature = torch.cat(new_feature,dim = 0)
        x_new_indices = torch.cat(new_indices,dim = 0)
        x_new = spconv.SparseConvTensor(
            features=x_new_feature,
            indices=x_new_indices,
            spatial_shape=self.output_shape,
            batch_size=len(record_len)
        )

        regrouped_batch_dict.update({"encoded_spconv_tensor": x_new})
        regrouped_batch_dict.update({"vote_loss": vote_loss})
        regrouped_batch_dict.update({"comm_points": comm_points})
        # print(regrouped_batch_dict)
        return regrouped_batch_dict

    def voting_info_confirm(self, x_voter, record_len):
        x_voter_feats = x_voter.features
        x_voter_indices = x_voter.indices
        x_voteprob = self.win_prob(x_voter_feats)

        # sort_voteprob, prob_indices = x_voteprob.view(-1,).sort()
        # x_voteprob = sort_voteprob
        # x_voter_feats = x_voter_feats[prob_indices]
        # x_voter_indices = x_voter_indices[prob_indices]

        loss_vote = torch.zeros(1).to(x_voteprob.device)
        cum_sum_len = torch.cumsum(record_len, dim=0)
        cum_sum_len = torch.cat([torch.tensor([0]).to(cum_sum_len.device), cum_sum_len], dim=0)
        kept_feature = []
        kept_indices = []
        comm_points = 0
        for ego_batch_id in range(record_len.shape[0]):
            egovoter = x_voteprob[x_voter_indices[:,0]==cum_sum_len[ego_batch_id]]
            egovoter_feats = x_voter_feats[x_voter_indices[:,0]==cum_sum_len[ego_batch_id]]
            egovoter_pos = x_voter_indices[x_voter_indices[:,0]==cum_sum_len[ego_batch_id]]
            kept_feature.append(egovoter_feats * egovoter)
            kept_indices.append(egovoter_pos)
            
            if record_len[ego_batch_id] > 1:
                colvoter = x_voteprob[(x_voter_indices[:, 0] >= (cum_sum_len[ego_batch_id]+1)) \
                                    & (x_voter_indices[:, 0] < cum_sum_len[ego_batch_id+1])]
                colvoter_feats = x_voter_feats[(x_voter_indices[:, 0] >= (cum_sum_len[ego_batch_id]+1)) \
                                    & (x_voter_indices[:, 0] < cum_sum_len[ego_batch_id+1])]
                colvoter_pos = x_voter_indices[(x_voter_indices[:, 0] >= (cum_sum_len[ego_batch_id]+1)) \
                                    & (x_voter_indices[:, 0] < cum_sum_len[ego_batch_id+1])]
                num_kept_col_windows = int(colvoter.shape[0]*(1-self.pruning_ratio))
                if not self.training:
                    comm_points = comm_points + num_kept_col_windows
                _, topk_indices = torch.topk(colvoter.view(-1),num_kept_col_windows)
                kept_colvoter = colvoter[topk_indices]
                kept_colvoter_feats = colvoter_feats[topk_indices]
                kept_colvoter_pos = colvoter_pos[topk_indices]
                if self.training:
                    loss_vote = loss_vote + self.weighted_chamfer_loss(egovoter,egovoter_pos[:,1:],kept_colvoter,kept_colvoter_pos[:,1:])
                kept_feature.append(kept_colvoter_feats * kept_colvoter)
                kept_indices.append(kept_colvoter_pos)
            else:
                if self.training:
                    loss_vote = loss_vote + self.weighted_chamfer_loss(egovoter,egovoter_pos[:,1:],egovoter,egovoter_pos[:,1:])
        x_kept_feature = torch.cat(kept_feature,dim = 0)
        x_kept_indices = torch.cat(kept_indices,dim = 0)
        x_kept = replace_feature(x_voter,x_kept_feature)
        x_kept.indices = x_kept_indices  
        # spconv.SparseConvTensor(
        #     features=x_kept_feature,
        #     indices=x_kept_indices,
        #     spatial_shape=x_voter.spatial_shape,
        #     batch_size=x_voter.batch_size
        #     
        if comm_points != 0: 
            comm_points = torch.log(torch.tensor(comm_points))
        else: comm_points = torch.tensor(comm_points)
        return x_kept,loss_vote,comm_points
    

    def weighted_chamfer_loss(self, egovoter, egovoter_pos, colvotor, colvoter_pos):
        conf1 = egovoter.squeeze(1)
        conf2 = 1 - colvotor.squeeze(1)
        pos1 = egovoter_pos.float()
        pos2 = colvoter_pos.float()
        ny, nx = self.output_shape
        winy = ny // 8
        winx = nx // 8
        pos1[:,0] = pos1[:,0]/winy
        pos2[:,0] = pos2[:,0]/winy
        pos1[:,1] = pos1[:,1]/winx
        pos2[:,1] = pos2[:,1]/winx
        
        dist = torch.cdist(pos1, pos2, p=2)
        weight = (conf1.unsqueeze(1) + conf2.unsqueeze(0)) / 2
        weighted_dist = dist * weight
        
        loss = weighted_dist.min(dim=1)[0].mean() + weighted_dist.min(dim=0)[0].mean()
        return loss
