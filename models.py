import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

class ConvE(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(ConvE, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels, 
                            (self.filt_h, self.filt_w), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        fc_length = (20-self.filt_h+1)*(20-self.filt_w+1)*self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)


    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 10, 20)
        r = self.R(r_idx).view(-1, 1, 10, 20)
        x = torch.cat([e1, r], 2)
        x = self.bn0(x)
        x= self.inp_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        return pred


class HypER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(HypER, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]

        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        fc_length = (1-self.filt_h+1)*(d1-self.filt_w+1)*self.out_channels
        self.fc = torch.nn.Linear(fc_length, d1)
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        self.fc1 = torch.nn.Linear(d2, fc1_length)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)


    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 1, self.E.weight.size(1))
        r = self.R(r_idx)
        x = self.bn0(e1)
        x = self.inp_drop(x)

        k = self.fc1(r)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e1.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e1.size(0))
        x = x.view(e1.size(0), 1, self.out_channels, 1-self.filt_h+1, e1.size(3)-self.filt_w+1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
       
        x = self.bn1(x)
        x = self.feature_map_drop(x) 
        x = x.view(e1.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x) 
        x = torch.mm(x, self.E.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        return pred


class DeepER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(DeepER, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]
        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()
        self.conv1 = P4ConvZ2(self.in_channels, self.out_channels, 3, 1, 0)
        self.conv2 = P4MConvZ2(self.in_channels, self.out_channels, 3, 1, 0)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        self.d3 = self.out_channels * 180  # For kinship dataset, use self.out_channels * 80
        fc_length = (1 - self.filt_h + 1) * (self.d3 - self.filt_w + 1) * self.out_channels * 4
        self.fc = torch.nn.Linear(fc_length, d1)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc0 = torch.nn.Linear(self.d3, fc1_length)
        self.fc1 = torch.nn.Linear(self.d3, fc1_length)
        self.fc2 = torch.nn.Linear(self.d3, fc1_length)
        self.fc3 = torch.nn.Linear(self.d3, fc1_length)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def head_reshape(self, x, b, i=1):
        x0 = x[0].reshape(b, i, i, -1)
        x1 = x[1].reshape(b, i, i, -1)
        x2 = x[2].reshape(b, i, i, -1)
        x3 = x[3].reshape(b, i, i, -1)
        return x0, x1, x2, x3

    def patches(self, group2, b, dim=2):
        u = torch.split(group2, 1, dim)
        u1 = u[0].reshape(b, -1)
        u2 = u[1].reshape(b, -1)
        u3 = u[2].reshape(b, -1)
        u4 = u[3].reshape(b, -1)
        return u1, u2, u3, u4

    def filters(self, k, b):
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(b * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)
        return k

    def permute(self, x0, x1, x2, x3):
        x0 = x0.permute(1, 0, 2, 3)
        x1 = x1.permute(1, 0, 2, 3)
        x2 = x2.permute(1, 0, 2, 3)
        x3 = x3.permute(1, 0, 2, 3)
        return x0, x1, x2, x3

    def convolve(self, y, b, i):
        y = y.view(b, 1, self.out_channels, 1 - self.filt_h + 1, i - self.filt_w + 1)
        y = y.permute(0, 3, 4, 1, 2)
        y = torch.sum(y, dim=3)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.bn1(y)
        y = self.feature_map_drop(y)
        return y

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 10, 20)
        r = self.R(r_idx).view(-1, 1, 10, 20)
        x = self.bn0(e1)
        x = self.inp_drop(x)
        x = self.conv1(x)
        r = self.conv2(r)
        r = torch.split(r, 4, dim=2)
        group1 = r[0]
        group2 = r[1]

        u1, u2, u3, u4 = self.patches(group2, e1.size(0))
        k0 = self.filters(self.fc0(u1), e1.size(0))
        k1 = self.filters(self.fc1(u2), e1.size(0))
        k2 = self.filters(self.fc2(u3), e1.size(0))
        k3 = self.filters(self.fc3(u4), e1.size(0))

        x = x + group1
        x = torch.split(x, 1, dim=2)
        x0, x1, x2, x3 = self.head_reshape(x, e1.size(0))
        x0, x1, x2, x3 = self.permute(x0, x1, x2, x3)
        y0 = self.convolve(F.conv2d(x0, k0, groups=e1.size(0)), e1.size(0), x0.size(3))
        y1 = self.convolve(F.conv2d(x1, k1, groups=e1.size(0)), e1.size(0), x1.size(3))
        y2 = self.convolve(F.conv2d(x2, k2, groups=e1.size(0)), e1.size(0), x2.size(3))
        y3 = self.convolve(F.conv2d(x3, k3, groups=e1.size(0)), e1.size(0), x3.size(3))

        y_concat = torch.cat([y0, y1, y2, y3], 2)
        y_concat = y_concat.view(e1.size(0), -1)
        y_concat = self.fc(y_concat)
        y_concat = self.hidden_drop(y_concat)
        y_concat = self.bn2(y_concat)
        y_concat = F.relu(y_concat)
        y_concat = torch.mm(y_concat, self.E.weight.transpose(1, 0))
        y_concat += self.b.expand_as(y_concat)
        pred = F.sigmoid(y_concat)
        return pred

class MDeepER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(MDeepER, self).__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_h = kwargs["filt_h"]
        self.filt_w = kwargs["filt_w"]
        self.d = d
        self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])
        self.loss = torch.nn.BCELoss()
        self.conv1 = P4ConvZ2(self.in_channels, self.out_channels, 3, 1, 0)
        self.conv2 = P4MConvZ2(self.in_channels, self.out_channels, 3, 1, 0)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(d1)
        self.register_parameter('b', Parameter(torch.zeros(len(d.entities))))
        self.d3 = self.out_channels * 204 # For WN-19 dataset, use self.out_channels * 204
        fc_length = (1 - self.filt_h + 1) * (self.d3 - self.filt_w + 1) * self.out_channels * 4
        self.fc = torch.nn.Linear(fc_length, d1)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc0 = torch.nn.Linear(self.d3, fc1_length)
        self.fc1 = torch.nn.Linear(self.d3, fc1_length)
        self.fc2 = torch.nn.Linear(self.d3, fc1_length)
        self.fc3 = torch.nn.Linear(self.d3, fc1_length)

    def init(self):
        self.E.weight.data = self.d.mul_emb
        xavier_normal_(self.R.weight.data)

    def head_reshape(self, x, b, i=1):
        x0 = x[0].reshape(b, i, i, -1)
        x1 = x[1].reshape(b, i, i, -1)
        x2 = x[2].reshape(b, i, i, -1)
        x3 = x[3].reshape(b, i, i, -1)
        return x0, x1, x2, x3

    def patches(self, group2, b, dim=2):
        u = torch.split(group2, 1, dim)
        u1 = u[0].reshape(b, -1)
        u2 = u[1].reshape(b, -1)
        u3 = u[2].reshape(b, -1)
        u4 = u[3].reshape(b, -1)
        return u1, u2, u3, u4

    def filters(self, k, b):
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(b * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)
        return k

    def permute(self, x0, x1, x2, x3):
        x0 = x0.permute(1, 0, 2, 3)
        x1 = x1.permute(1, 0, 2, 3)
        x2 = x2.permute(1, 0, 2, 3)
        x3 = x3.permute(1, 0, 2, 3)
        return x0, x1, x2, x3

    def convolve(self, y, b, i):
        y = y.view(b, 1, self.out_channels, 1 - self.filt_h + 1, i - self.filt_w + 1)
        y = y.permute(0, 3, 4, 1, 2)
        y = torch.sum(y, dim=3)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.bn1(y)
        y = self.feature_map_drop(y)
        return y

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx).view(-1, 1, 12, 19)
        r = self.R(r_idx).view(-1, 1, 12, 19)
        x = self.bn0(e1)
        x = self.inp_drop(x)
        x = self.conv1(x)
        r = self.conv2(r)
        r = torch.split(r, 4, dim=2)
        group1 = r[0]
        group2 = r[1]

        u1, u2, u3, u4 = self.patches(group2, e1.size(0))
        k0 = self.filters(self.fc0(u1), e1.size(0))
        k1 = self.filters(self.fc1(u2), e1.size(0))
        k2 = self.filters(self.fc2(u3), e1.size(0))
        k3 = self.filters(self.fc3(u4), e1.size(0))

        x = x + group1
        x = torch.split(x, 1, dim=2)
        x0, x1, x2, x3 = self.head_reshape(x, e1.size(0))
        x0, x1, x2, x3 = self.permute(x0, x1, x2, x3)
        y0 = self.convolve(F.conv2d(x0, k0, groups=e1.size(0)), e1.size(0), x0.size(3))
        y1 = self.convolve(F.conv2d(x1, k1, groups=e1.size(0)), e1.size(0), x1.size(3))
        y2 = self.convolve(F.conv2d(x2, k2, groups=e1.size(0)), e1.size(0), x2.size(3))
        y3 = self.convolve(F.conv2d(x3, k3, groups=e1.size(0)), e1.size(0), x3.size(3))

        y_concat = torch.cat([y0, y1, y2, y3], 2)
        y_concat = y_concat.view(e1.size(0), -1)
        y_concat = self.fc(y_concat)
        y_concat = self.hidden_drop(y_concat)
        y_concat = self.bn2(y_concat)
        y_concat = F.relu(y_concat)
        y_concat = torch.mm(y_concat, self.E.weight.transpose(1, 0))
        y_concat += self.b.expand_as(y_concat)
        pred = F.sigmoid(y_concat)
        return pred

