import torch.nn as nn
import torch.nn.functional as F
import torch
class MLP(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

#Alternate Mapper if not tune gpt
###############################################
class MultiHeadAttention(nn.Module):
  def __init__(self, dim_self, num_heads, bias=True, dropout=0.0):
    super().__init__()
    self.num_heads = num_heads
    head_dim = dim_self//num_heads
    self.scale = head_dim ** -0.5
    self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
    self.to_keys_values = nn.Linear(dim_self, dim_self*2, bias=bias)
    self.project = nn.Linear(dim_self, dim_self)
  def forward(self, x):
    b,n,c = x.shape
    queries = self.to_queries(x).reshape(b,n,self.num_heads,c // self.num_heads)
    k,v = (self.to_keys_values(x)).split(2,dim=-1)
    k = k.reshape(b,n,self.num_heads,c // self.num_heads)
    v = v.reshape(b,n,self.num_heads,c // self.num_heads)
    att = torch.einsum('bnhd,bmhd->bnmh', queries, k) * self.scale
    att = att.softmax(dim=2)
    out = torch.einsum('bnmh,bmhd->bnhd',att,v).reshape(b,n,c)
    return out

class MLPB(nn.Module):
  def __init__(self, in_dim, h_dim, out_dim=None,act=F.relu,dropout=0.0):
    super().__init__()
    out_d = out_dim if out_dim is not None else in_dim
    self.fc1 = nn.Linear(in_dim, h_dim)
    self.act = act
    self.fc2 = nn.Linear(h_dim, out_d)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x

class Block(nn.Module):
  def __init__(self,dim_self,num_heads, mlp_ratio=4,bias=False,dropout=0.0,act=F.relu,norm_layer=nn.LayerNorm):
    super().__init__()
    self.norm1 = norm_layer(dim_self)
    self.attn = MultiHeadAttention(dim_self,num_heads,bias=bias,dropout=dropout)
    self.norm2 = norm_layer(dim_self)
    self.mlp = MLPB(dim_self,int(dim_self*mlp_ratio),act=act,dropout=dropout)
  def forward(self,x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

class Transformer(nn.Module):
  def __init__(self,dim_self,num_heads, num_layers):
    super().__init__()
    layers = []
    for i in range(num_layers):
      layers.append(Block(dim_self,num_heads))
    self.layers = nn.Sequential(*layers)
  def forward(self,x):
    x = self.layers(x)
    return x

class Mapper(nn.Module):
  def __init__(self,dim_clip, dim_embedding,prefix_length,clip_length, num_layers=8):
    super().__init__()
    self.clip_length = clip_length
    self.transformer = Transformer(dim_embedding, 8, num_layers)
    self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
    self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
  def forward(self,x):
    x = self.linear(x).view(x.shape[0], self.clip_length, -1)
    prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
    prefix = torch.cat((x,prefix), dim=1)
    out = self.transformer(prefix)[:, self.clip_length:]
    return out