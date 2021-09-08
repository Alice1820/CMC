import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        # K = int(self.params[0].item())
        K = self.K
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab

def protoSimilarity(x, weight_y, weight_z, K=127, batchSize=16, inputSize=128):
    # ===================view x=====================
    pos_x = x.unsqueeze(1).contiguous().repeat(1, (K+1)^2, 1) # [bs, 1, 128] => [bs, (K+1)^2, 128]
    feat_y = weight_y.unsqueeze(1).repeat(1, K+1, 1, 1)
    feat_y = feat_y.contiguous().view(batchSize, (K+1)^2, inputSize) # [bs, K+1, K+1, 128] => [bs, (K+1)^2, 128]
    feat_z = weight_z.unsqueeze(2).repeat(1, 1, K+1, 1)
    feat_z = feat_z.contiguous().view(batchSize, (K+1)^2, inputSize) # [bs, K+1, K+1, 128] => [bs, (K+1)^2, 128]
    center = (pos_x + feat_y + feat_z) / 3 # [bs, (K+1)^2, 128]
    del_x = torch.norm(center - out_x, p=2, dim=-1) # # [bs, (K+1)^2]
    out_x = del_x * (-1)
    return out_x

class NCEAverageXYZ(nn.Module):

    def __init__(self, inputSize=128, outputSize=37800, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverageXYZ, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_x', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_y', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_z', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, z, index):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_x = self.params[2].item()
        Z_y = self.params[3].item()
        Z_z = self.params[4].item()

        momentum = self.params[5].item()
        batchSize = x.size(0)
        outputSize = self.memory_x.size(0)
        inputSize = self.memory_x.size(1)

        # x: [bs, 128]
        # y: [bs, 128]
        # z: [bs, 128]
        # score computation
        idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
        idx.select(1, 0).copy_(index.data)
        # sample
        weight_x = torch.index_select(self.memory_x, 0, idx.view(-1)).detach()
        weight_x = weight_x.view(batchSize, K + 1, inputSize)
        # sample
        weight_y = torch.index_select(self.memory_y, 0, idx.view(-1)).detach()
        weight_y = weight_y.view(batchSize, K + 1, inputSize)
        # sample
        weight_z = torch.index_select(self.memory_z, 0, idx.view(-1)).detach()
        weight_z = weight_z.view(batchSize, K + 1, inputSize)

        if self.use_softmax:
            out_x = protoSimilarity(x, weight_y, weight_z, K, batchSize, inputSize)
            out_y = protoSimilarity(y, weight_x, weight_z, K, batchSize, inputSize)
            out_z = protoSimilarity(z, weight_x, weight_y, K, batchSize, inputSize) # [bs, (K+1)^2]

            out_x = torch.div(out_x, T)
            out_y = torch.div(out_y, T)
            out_z = torch.div(out_z, T)
            out_x = out_x.contiguous()
            out_y = out_y.contiguous()
            out_z = out_z.contiguous()
            
        else:
            raise Exception("3-view NCE loss not implemented.")

        # # update memory
        with torch.no_grad():
            x_pos = torch.index_select(self.memory_x, 0, index.view(-1))
            x_pos.mul_(momentum)
            x_pos.add_(torch.mul(l, 1 - momentum))
            x_norm = x_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_x = x_pos.div(x_norm)
            self.memory_x.index_copy_(0, index, updated_x)

            y_pos = torch.index_select(self.memory_y, 0, index.view(-1))
            y_pos.mul_(momentum)
            y_pos.add_(torch.mul(y, 1 - momentum))
            y_norm = y_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_y = y_pos.div(y_norm)
            self.memory_y.index_copy_(0, index, updated_y)

            z_pos = torch.index_select(self.memory_z, 0, index.view(-1))
            z_pos.mul_(momentum)
            z_pos.add_(torch.mul(z, 1 - momentum))
            z_norm = z_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_z = z_pos.div(z_norm)
            self.memory_z.index_copy_(0, index, updated_z)

        return out_x, out_y, out_z

# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out
