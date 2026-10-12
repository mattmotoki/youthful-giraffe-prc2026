"""Input-only 17-token mixed-runway transformer and matched row control."""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self,n_numeric,vocab_sizes,arm):
        super().__init__()
        if arm not in ('sequence','row'):raise ValueError('Unknown arm')
        self.arm=arm
        self.embeds=nn.ModuleList([nn.Embedding(v,8) for v in vocab_sizes])
        if arm=='sequence':
            self.numeric=nn.Linear(n_numeric,64)
            self.category=nn.Linear(8*len(vocab_sizes),64)
            self.elapsed=nn.Linear(1,64)
            layer=nn.TransformerEncoderLayer(64,4,128,dropout=.1,batch_first=True)
            self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.head=nn.Linear(64,1)
        else:
            self.head=nn.Sequential(nn.Linear(n_numeric+8*len(vocab_sizes),64),nn.ReLU(),
                                    nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,numeric,category,elapsed,padding):
        cats=torch.cat([e(category[:,:,j]) for j,e in enumerate(self.embeds)],dim=-1)
        if self.arm=='sequence':
            if numeric.shape[1]!=17 or padding[:,8].any():raise ValueError('Invalid centre/padding')
            h=self.numeric(numeric)+self.category(cats)+self.elapsed(elapsed[:,:,None]/300.)
            h=self.context(h,src_key_padding_mask=padding)
            return self.head(h[:,8]).squeeze(-1)
        if numeric.shape[1]!=1:raise ValueError('Row control received neighbours')
        return self.head(torch.cat([numeric[:,0],cats[:,0]],dim=-1)).squeeze(-1)
