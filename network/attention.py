import torch
import torch.nn as nn


class PAM(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(PAM, self).__init__()
        in_channels = in_channels
        num_filters = in_channels // reduction
        self.query_conv = nn.Conv2d(in_channels, num_filters, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels, num_filters, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        att = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, att.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = x.view(b, c, -1)
        proj_key = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        att = self.softmax(energy)
        proj_value = x.view(b, c, -1)
        out = torch.bmm(att, proj_value)
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class DAM(nn.Module):
    def __init__(self, in_channels, num_filters, out_channels, reduction=8):
        super(DAM, self).__init__()
        self.pa_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.SELU(inplace=True)
        )
        self.ca_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.SELU(inplace=True)
        )
        self.pa = PAM(in_channels, reduction)
        self.ca = CAM()
        self.sum_conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(num_filters, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x):
        p_out = self.pa_conv(x)
        c_out = self.ca_conv(x)
        p_out = self.pa(p_out)
        c_out = self.ca(c_out)
        s_out = p_out + c_out
        out = self.sum_conv(s_out)
        return out
