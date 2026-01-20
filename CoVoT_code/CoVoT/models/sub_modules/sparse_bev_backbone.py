import spconv.pytorch as spconv
import torch
import torch.nn as nn


class SparseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if 'layer_nums' in self.model_cfg:
            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                spconv.SparseConv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=1, bias=False
                ),
                spconv.SparseBatchNorm(num_filters[idx]),
                spconv.SparseReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    spconv.SubMConv2d(
                        num_filters[idx], num_filters[idx],
                        kernel_size=3, padding=1, bias=False
                    ),
                    spconv.SparseBatchNorm(num_filters[idx]),
                    spconv.SparseReLU()
                ])

            self.blocks.append(spconv.SparseSequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        spconv.SparseConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            kernel_size=upsample_strides[idx],
                            stride=upsample_strides[idx],
                            bias=False
                        ),
                        spconv.SparseBatchNorm(num_upsample_filters[idx],
                                    eps=1e-3, momentum=0.01),
                        spconv.SparseReLU()
                    ))
                else:
                    stride = round(1 / stride)
                    self.deblocks.append(nn.Sequential(
                    spconv.SubMConv2d(
                        num_filters[idx], num_upsample_filters[idx],
                        kernel_size=stride, stride=stride, bias=False
                    ),
                    spconv.SparseBatchNorm(num_upsample_filters[idx],
                                    eps=1e-3, momentum=0.01),
                    spconv.SparseReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
            spconv.SparseConvTranspose2d(
                c_in, c_in, kernel_size=upsample_strides[-1],
                stride=upsample_strides[-1], bias=False
            ),
            spconv.SparseBatchNorm(c_in,eps=1e-3, momentum=0.01),
            spconv.SparseReLU()))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['encoded_spconv_tensor']  # 稀疏张量

        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.spatial_shape[0] / x.spatial_shape[0])
            ret_dict['spatial_features_%dx' % stride] = x
            # print('spatial_features_shape:', x.features.shape)

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x).dense())
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict
