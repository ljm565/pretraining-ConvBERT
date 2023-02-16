import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils_model import DepthWiseSeparableConv1d



# word embedding layer
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output



# positional embedding layer
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, device):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.device = device

        self.pos = torch.arange(0, self.max_len)
        self.emb_layer = nn.Embedding(self.max_len, self.hidden_dim)


    def forward(self, x):
        return self.emb_layer(self.pos.unsqueeze(0).to(self.device))[:, :x.size(1)]



# segment embedding layer
class SegmentEmbedding(nn.Module):
    def __init__(self, hidden_dim, pad_token_id):
        super(SegmentEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(3, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output
        


# self attention
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, reduced_hidden_dim, head_dim, bias, self_attn, causal):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.reduced_hidden_dim = reduced_hidden_dim
        self.head_dim = head_dim
        self.num_head = self.reduced_hidden_dim // self.head_dim
        self.bias = bias
        self.self_attn = self_attn
        self.causal = causal

        self.attn_proj = nn.Linear(self.reduced_hidden_dim, self.hidden_dim//2, bias=self.bias)


    def head_split(self, x):
        x = x.view(self.batch_size, -1, self.num_head, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x


    def scaled_dot_product(self, q, k, v, mask):
        attn_wts = torch.matmul(q, k.transpose(2, 3))/(self.head_dim ** 0.5)
        if not mask == None:
            attn_wts = attn_wts.masked_fill(mask==0, float('-inf'))
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_out = torch.matmul(attn_wts, v)
        return attn_wts, attn_out


    def reshaping(self, attn_out):
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(self.batch_size, -1, self.reduced_hidden_dim)
        return attn_out


    def forward(self, q, k, v, mask):
        self.batch_size = q.size(0)

        q, k, v = self.head_split(q), self.head_split(k), self.head_split(v)
        attn_wts, attn_out = self.scaled_dot_product(q, k, v, mask)
        attn_out = self.attn_proj(self.reshaping(attn_out))

        return attn_wts, attn_out



# span-based dynamic convolutional attention
class SpanDynamicConvAttention(nn.Module):
    def __init__(self, hidden_dim, reduced_hidden_dim, head_dim, kernel_size, num_head, bias, dynamic):
        super(SpanDynamicConvAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.reduced_hidden_dim = reduced_hidden_dim
        self.kernel_size = kernel_size
        self.num_head = num_head
        self.head_dim = head_dim
        self.bias = bias
        self.dynamic = dynamic
        
        if self.dynamic:
            self.dynamic_kernel_layer = nn.Linear(self.reduced_hidden_dim, self.kernel_size * self.num_head, bias=self.bias)   # not k but k*num_head due to depth-wise operation
        else:
            self.kernel = nn.Parameter(torch.ones(1, 1, self.num_head, self.kernel_size, 1) * np.log(1 / 0.07))
        self.attn_proj = nn.Linear(self.reduced_hidden_dim, self.hidden_dim//2, bias=self.bias)
        
    
    def expand4conv(self, v):
        v = F.pad(v, (0, 0, self.kernel_size//2, self.kernel_size//2), 'constant', 0)   # B x L x D -> B x (L+2P) x D
        v = v.transpose(1, 2)   # B x (L+2P) x D -> B x D x (L+2P)
        expanded = [v[:, :, i:i+self.kernel_size].unsqueeze(-1) for i in range(v.size(-1)-self.kernel_size+1)]
        expanded = torch.cat(expanded, dim=-1)   # B x D x k x L
        expanded = torch.permute(expanded, (0, 3, 1, 2))   # B x L x D x k
        expanded = expanded.view(self.batch_size, -1, self.num_head, self.head_dim, self.kernel_size)    # B x L x num_head x head_dim x k
        return expanded


    def forward(self, q, ks, v):
        self.batch_size = q.size(0)

        v = self.expand4conv(v)   # B x L x num_head x head_dim x k

        if self.dynamic:
            dynamic_kernel = self.dynamic_kernel_layer(torch.mul(q, ks))   # B x L x (num_head x k)
            # dynamic_kernel = F.softmax(dynamic_kernel.view(self.batch_size, -1, self.num_head, self.kernel_size), dim=-1).unsqueeze(-1)   # B x L x num_head x k x 1
            dynamic_kernel = F.softmax(dynamic_kernel.view(self.batch_size, -1, self.num_head, self.kernel_size, 1), dim=3)   # B x L x num_head x k x 1

            output = torch.matmul(v, dynamic_kernel).view(self.batch_size, -1, self.reduced_hidden_dim)
            output = self.attn_proj(output)
        
        else:
            output = torch.matmul(v, self.kernel).view(self.batch_size, -1, self.reduced_hidden_dim)
            output = self.attn_proj(output)

        return output



# mixed attention
class MixedAttention(nn.Module):
    def __init__(self, config, self_attn, causal):
        super(MixedAttention, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_head
        self.kernel_size = config.kernel_size
        self.bias = bool(config.bias)
        self.gamma = config.gamma
        self.self_attn = self_attn
        self.causal = causal
        self.dynamic = config.dynamic

        if not self.hidden_dim % self.num_head == 0:
            raise ValueError('hidden dimension must be divided into number of heads')
        if not self.kernel_size % 2 == 1:
            raise ValueError('convolution kenrel size must be a odd number due to depth-wise separable convolution')

        # reduce feature for computational cost
        self.head_dim = self.hidden_dim // self.num_head   # original head dimension
        self.num_head = self.num_head // self.gamma
        if self.num_head < 1:
            self.num_head = 1
            self.gamma = config.num_head
        self.reduced_hidden_dim = self.head_dim * self.num_head

        self.q_proj = nn.Linear(self.hidden_dim, self.reduced_hidden_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.reduced_hidden_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.reduced_hidden_dim, bias=self.bias)
        self.ks_proj = DepthWiseSeparableConv1d(
            in_channels=self.hidden_dim,
            out_channels=self.reduced_hidden_dim,
            kernel_size=self.kernel_size,
            bias=self.bias)   # to reduce computational cost

        self.selfAttention = SelfAttention(self.hidden_dim, self.reduced_hidden_dim, self.head_dim, self.bias, self.self_attn, self.causal)
        self.dynamicAttention = SpanDynamicConvAttention(self.hidden_dim, self.reduced_hidden_dim, self.head_dim, self.kernel_size, self.num_head, self.bias, self.dynamic)


    def forward(self, query, key, value, mask):
        if self.self_attn:
            assert (query == key).all() and (key==value).all()

        self.batch_size = query.size(0)

        q = self.q_proj(query)
        v = self.v_proj(value)
        k = self.k_proj(key)
        ks = self.ks_proj(key.transpose(1, 2)).transpose(1, 2)

        attn_score, self_attn_output = self.selfAttention(q, k, v, mask)
        dynamic_attn_output = self.dynamicAttention(q, ks, v)
        attn_output = torch.cat((self_attn_output, dynamic_attn_output), dim=-1)
        
        return attn_score, attn_output



# grouped feed forward network
class GroupedFFN(nn.Module):
    def __init__(self, config):
        super(GroupedFFN, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.max_len = config.max_len
        self.ffn_dim = config.ffn_dim
        self.group = config.group
        self.dropout = config.dropout
        self.bias = bool(config.bias)

        if self.hidden_dim % self.group != 0 or self.ffn_dim % self.group != 0:
            raise ValueError('hidden and intermediate dimension must be divided into group number')

        self.in_dim = self.hidden_dim // self.group
        self.mid_dim = self.ffn_dim // self.group

        self.FFN1 = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim, bias=self.bias),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(self.mid_dim, self.in_dim, bias=self.bias),
        )
        self.init_weights()


    def init_weights(self):
        for _, param in self.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param.data, mean=0, std=0.02)

    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.max_len, self.group, -1)
        x = self.FFN1(x)
        x = self.FFN2(x)
        x = x.view(batch_size, self.max_len, -1)
        return x



# a single encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.config = config
        self.hidden_dim = self.config.hidden_dim
        self.dropout = self.config.dropout
        self.layernorm_eps = config.layernorm_eps

        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.self_attention = MixedAttention(config, self_attn=True, causal=False)
        self.positionWiseFeedForward = GroupedFFN(self.config)


    def forward(self, x, mask):
        attn_score, output = self.self_attention(query=x, key=x, value=x, mask=mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        output = self.positionWiseFeedForward(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        return attn_score, output



# all encoders
class Encoder(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Encoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.device = device

        self.config = config
        self.num_layers = self.config.num_layers
        self.hidden_dim = self.config.hidden_dim
        self.dropout = self.config.dropout
        self.max_len = config.max_len
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.tok_emb = TokenEmbedding(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_emb = PositionalEmbedding(self.max_len, self.hidden_dim, self.device)
        self.seg_emb = SegmentEmbedding(self.hidden_dim, self.pad_token_id)
        self.encoders = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.num_layers)])


    def forward(self, x, segment, mask=None):
        output = self.tok_emb(x) + self.pos_emb(x) + self.seg_emb(segment)
        output = self.dropout_layer(output)

        all_attn_wts = []
        for encoder in self.encoders:
            attn_wts, output = encoder(output, mask)
            all_attn_wts.append(attn_wts.detach().cpu())
        
        return all_attn_wts, output


# ConvBERT
class ConvBERT(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(ConvBERT, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        self.hidden_dim = self.config.hidden_dim

        self.encoder = Encoder(self.config, self.tokenizer, self.device)
        self.nsp_fc = nn.Linear(self.hidden_dim, 2)
        self.mlm_fc = nn.Linear(self.hidden_dim, self.tokenizer.vocab_size)


    def make_mask(self, input):
        enc_mask = torch.where(input==self.tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        return enc_mask


    def forward(self, x, segment):
        enc_mask = self.make_mask(x)
        all_attn_wts, x = self.encoder(x, segment, enc_mask)

        nsp_output = self.nsp_fc(x[:, 0])
        mlm_output = self.mlm_fc(x)

        return all_attn_wts, (nsp_output, mlm_output)