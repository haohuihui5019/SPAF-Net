import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
from model.FSAC import FSACblock
from model.CBAM import CBAM


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True,
                 bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SCAModule(nn.Module):
    def __init__(self, inn, out):
        super(SCAModule, self).__init__()
        base = inn // 4
        self.conv_sa = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bias=False),
                                     SAM(base),
                                     Conv2d(base, base, 3, same_padding=True, bias=False)
                                     )
        self.conv_ca = nn.Sequential(Conv2d(inn, base, 3, same_padding=True, bias=False),
                                     CAM(base),
                                     Conv2d(base, base, 3, same_padding=True, bias=False)
                                     )
        self.conv_cat = Conv2d(base * 2, out, 1, same_padding=True, bn=False)

    def forward(self, x):
        sa_feat = self.conv_sa(x)
        ca_feat = self.conv_ca(x)
        cat_feat = torch.cat((sa_feat, ca_feat), 1)
        cat_feat = self.conv_cat(cat_feat)
        return cat_feat


class SAM(nn.Module):
    def __init__(self, channel):
        super(SAM, self).__init__()
        self.para_lambda = nn.Parameter(torch.zeros(1))
        self.query_conv = Conv2d(channel, channel // 8, 1, NL='none')
        self.key_conv = Conv2d(channel, channel // 8, 1, NL='none')
        self.value_conv = Conv2d(channel, channel, 1, NL='none')

    def forward(self, x):
        N, C, H, W = x.size()
        proj_query = self.query_conv(x).view(N, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(N, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(N, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)

        out = self.para_lambda * out + x
        return out


class CAM(nn.Module):
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.para_mu = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, C, H, W = x.size()
        proj_query = x.view(N, C, -1)
        proj_key = x.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy, dim=-1)
        proj_value = x.view(N, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)

        out = self.para_mu * out + x
        return out
class VGG(nn.Module):
    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        self.upsample1 = Upsample(512, 256, 256 + 512, 512)
        self.upsample2 = Upsample(512, 256, 256 + 256, 512)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.lsk = FSACblock(512)
        self.cbam = CBAM(512)

        self.cls_head = nn.Sequential(nn.Conv2d(512, 512, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, num_classes, 1, 1))

        self.reg_head = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 128, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 64, 3, 1, 1),
                                      nn.Conv2d(64, 3, 3, 1, 1))
        self.apply(self._init_weights)
        self.output_layer = SCAModule(64, 3)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, x):
        x1 = self.features[: 19](x) #256 16
        x2 = self.features[19: 28](x1)  # 512 8
        x3 = self.features[28:](x2)  # 512 4

        feat = self.upsample1(x3, x2)
        feat = self.upsample2(feat, x1)
        feat = self.cbam(feat)
        feat, attn = self.lsk(feat)
        pred_den = self.reg_head(feat)

        return pred_den



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    print(nn.Sequential(*layers))
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class Upsample(nn.Module):
    def __init__(self, up_in_ch, up_out_ch, cat_in_ch, cat_out_ch):  # 512,256 512
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(*[nn.Conv2d(up_in_ch, up_out_ch, 3, 1, padding=1), nn.ReLU()])
        self.conv2 = nn.Sequential(*[nn.Conv2d(cat_in_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU(),
                                     nn.Conv2d(cat_out_ch, cat_out_ch, 3, 1, padding=1), nn.ReLU()])

    def forward(self, low, high):
        low = self.up(low)
        low = self.conv1(low)
        x = torch.cat([high, low], dim=1)

        x = self.conv2(x)
        return x



def vgg19(num_classes):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), num_classes)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
