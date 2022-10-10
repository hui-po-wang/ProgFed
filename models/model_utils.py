import torch
import torch.nn as nn

def get_outdim(dset):
    if dset == 'imagenet':
        return 1000
    elif dset == 'cifar100':
        return 100
    elif dset == 'cifar10' or dset == 'mnist':
        return 10
    else:
    	raise NotImplementedError()

class SingleSubModel(nn.Module):
    """Submodels that produce an only single output"""
    def __init__(self, enc, head, strategy, ind):
        super(SingleSubModel, self).__init__()
        m_list = nn.ModuleList()
        for m in enc:
            m_list.append(m)
        
        self.enc = m_list
        self.head = head
        self.strategy = strategy
        self.ind = ind

    def forward(self, x, verbose=False):
        feats = []
        out = x
        for m in self.enc:
            out = m(out)
            feats.append(out)
        
        #print(feats[-1].size())
        if not verbose:
            return self.head(feats[-1])
        else:
            return self.head(feats[-1]), feats

    def print_weight(self):
        for n, p in self.named_parameters():
            print(n, p)

    def trainable_parameters(self):
        for i in range(self.ind+1):
            for name, param in self.enc[i].named_parameters():
                yield param
        for name, param in self.head.named_parameters():
            yield param

    def lastest_parameters(self):
        for name, param in self.enc[self.ind].named_parameters():
            yield param

        if isinstance(self.head, list):
            for h in self.head:
                for name, param in h.named_parameters():
                    yield param
        else:
            for name, param in self.head.named_parameters():
                yield param

    def return_num_parameters(self):
        total = 0
        # Since enc and dec are module lists, we have to travese every model in them.
        for p in self.trainable_parameters():
            total += torch.numel(p)

        return total

class MultiSubModel(SingleSubModel):
    """Submodels that produce multiple outputs"""
    def __init__(self, enc, head, strategy, ind):
        super(MultiSubModel, self).__init__(enc, head, strategy, ind)

        m_list = nn.ModuleList()
        for m in enc:
            m_list.append(m)
        self.enc = m_list

        h_list = nn.ModuleList()
        for h in head:
            h_list.append(h)
        self.head = h_list

        self.strategy = strategy
        self.ind = ind

    def forward(self, x, verbose=False):
        feats = []
        outs = []

        feat = x
        for m_f, m_o in zip(self.enc, self.head):
            feat = m_f(feat)
            out = m_o(feat)

            outs.append(out)
            feats.append(feat)

        
        if not verbose:
            return outs
        else:
            return outs, feats
