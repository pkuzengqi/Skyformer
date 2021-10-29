import math
import time
import torch
from torch import nn
import torch.linalg as la

from functools import partial
from models.attention import SoftmaxAttention as SelfAttention
from config import Config
from torch.autograd import Function, gradcheck

def linear_attention(q, k, v): # for SM kernel
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))

    # temp = torch.einsum('...n,...nd,...md->...nm', D_inv, q, k)
    # print("Sketched SM:", la.norm(temp, ord=2, dim=(2,3)).mean())
    # del temp

    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out



def kernel_SM(X1, X2=None, X2_accu=False):
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...mnd', X1, X2)
        # product = torch.matmul(X1.unsqueeze(dim=2), torch.transpose(X2, 3, 4))
        result = torch.exp(product)

        result = result.sum(dim=2)
        # print(result.shape)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)

    return result
    # return result, product

def kernel_RS_SM(X1, X2=None, X2_accu=False, random_sign=None):
    # RS for random sign
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...mnd', X1, X2)
        # product = torch.matmul(X1.unsqueeze(dim=2), torch.transpose(X2, 3, 4))
        result = torch.exp(product)
        result = torch.transpose(result, 2, 3) # nmd
        result = result * random_sign
        result = result.sum(dim=3)

    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)

    return result
    
def kernel_RS_SM1(X1, X2=None, X2_accu=False, random_sign=None):
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...nmd', X1, X2)
        result = torch.exp(product)
        result = torch.einsum('bhnmd,...bmd->...bhnd', result, random_sign)
        # result = (result.transpose(0, 2) * random_sign).sum(-2).transpose(0, 2) # nhbmd -> nhbd -> bhnd

    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)

    return result
    # return result, product

def kernel_RS_RBF(X1, X2=None, X2_accu=False, random_sign=None):

    # todo

    if X2 is None:
        X2 = X1
        X2_accu = False

    diag_X1 = (X1 * X1).sum(-1) * 0.5
    diag_X1 = diag_X1.unsqueeze(dim=-1)
    diag_X2 = (X2 * X2).sum(-1) * 0.5
    diag_X2 = diag_X2.unsqueeze(dim=-2)

    if X2_accu:
        diag_X1 = diag_X1.unsqueeze(dim=-3)
        product = torch.einsum('...np,...mdp->...mnd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)
        result = torch.transpose(result, 2, 3) # nmd
        result = torch.einsum('bhnmd,bmd->bhnd', result, random_sign)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)

    return result

def rbf_attention(q, k, v): # for rbf kernel
    # todo
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

def kernel_sketch(q, k, *, kernel_fn, sketching_matrix, random_sign, normalize_data=False, eps=1e-4, device = None):
    # sketching_matrix: (self.M, self.d) tensor
    # sketching_matrix

    b, h, n, p = q.shape

    # data_normalizer = (p ** -0.25) if normalize_data else 1.
    # X = torch.cat([q, k], dim=2) * data_normalizer
    X = torch.cat([q, k], dim=2)
    
    XS = X.transpose(1, 2)[torch.arange(b)[:, None, None], sketching_matrix].permute(0,3,1,2,4) # bmdhp -> bhmdp
    AS = kernel_fn(X, XS, True, random_sign)
    # AS, p = kernel_fn(X, X[:,:,sketching_matrix], True)
    # if True in torch.isinf(AS):
    # print()
    # if True in torch.isnan(AS):
    # print()

    return AS.type_as(q)
        
def iterative_inv(mat, n_iter = 6, init_option = "original"):
    

    I = torch.eye(mat.size(-1), device = mat.device)
    K = mat
    
    if init_option == "original":
        V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
    else:
        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
    
    for _ in range(n_iter):
        # print(V)
    
        KV = torch.matmul(K, V)
        V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
    return V

 

# class SketchedAttentionRBF(nn.Module):
class Skyformer(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.device = config["device"] if "device" in config else "cuda"
        n = config["max_seq_len"]
        
        self.accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]

        # nb_features = default(nb_features, int(math.sqrt(n)))
        # nb_features = default(nb_features, int(3*n ** (1.0/4)))
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')


        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features

        if config["sketched_kernel"] == "kernel_RS_RBF":
            self.kernel_fn = kernel_RS_RBF

        self.no_projection = config["no_projection"]


    @torch.no_grad()
    def uniform_sketching(self, n, nb_rows, nb_columns, non_padding_num):
        
        total = nb_rows * nb_columns
        S = torch.rand(total, device=self.device)
        S = torch.einsum("b,d->bd", non_padding_num, S).long()
        S[:, total//2:] = S[:, total//2:] + n
        S = S.reshape(-1, nb_rows, nb_columns)
        # random_sign = (torch.randint(2, S.shape, device=self.device) * 2 - 1) * (math.sqrt(2 * n) / nb_rows / nb_columns)
        random_sign = torch.ones(S.shape, device=self.device)

        return S, random_sign

    def forward(self, q, k, v, mask):

        
        device = q.device
        b, h, n, d = q.shape
        
        data_normalizer = (32 ** -0.25)
        

        q = q * (mask[:, None, :, None] * data_normalizer)
        k = k * (mask[:, None, :, None] * data_normalizer)
        v = v * mask[:, None, :, None]
        

        
        non_padding_num = mask.sum(-1) # b


        self.sketching_matrix, self.random_sign = self.uniform_sketching(
            n, self.accumulation, self.nb_features, non_padding_num) # bmd
        

        create_kernel_sketch = partial(kernel_sketch, kernel_fn = self.kernel_fn,
           sketching_matrix = self.sketching_matrix, random_sign=self.random_sign, device = device)
        AS = create_kernel_sketch(q, k)  # b,h,2n, nb_feat
        Q = AS[:,:,:n] # b, h, n, nb_feat
        

        STAS = AS.transpose(1, 2)[torch.arange(b)[:, None, None], self.sketching_matrix] # bnhd -> bmdhd
        STAS = torch.einsum('bmdhe,bmd->bhde', STAS, self.random_sign) # bmdhd -> bhdd

        STAS = STAS + 1e-1*torch.eye(STAS.shape[-1], device=self.device)
 
        K = AS[:,:,n:]
        
        
        ##################################################################
        D_STAS_inv = 1 / STAS.sum(-1)
        D_STAS_inv = torch.sqrt( D_STAS_inv)
        STAS = torch.einsum("...d,...de,...e->...de", D_STAS_inv, STAS, D_STAS_inv)
        
        
        
        STAS_inv = iterative_inv(STAS, 6)
        
        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv) @ STAS_inv
        

        
        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv) * mask[:, None, :, None]
        
        ##################################################################

        context = torch.einsum('...nd,...ne->...de', K, v)
        out = torch.matmul(Q, context)    
        
        return out   
