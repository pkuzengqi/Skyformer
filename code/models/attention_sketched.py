import math
import time
import torch
from torch import nn
import torch.linalg as la

from functools import partial
from models.attention import SoftmaxAttention as SelfAttention
from config import Config
from torch.autograd import Function, gradcheck

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

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

def kernel_RBF(X1, X2=None, X2_accu=False):

    # todo

    if X2 is None:
        X2 = X1
        X2_accu = False

    diag_X1 = torch.abs(X1) ** power
    diag_X1 = torch.sum(diag_X1, dim=-1) / scale
    diag_X1 = diag_X1.unsqueeze(dim=-1)
    diag_X2 = torch.abs(X2) ** power
    diag_X2 = torch.sum(diag_X2, dim=-1) / scale
    diag_X2 = diag_X2.unsqueeze(dim=-2)

    if X2_accu:
        diag_X1 = diag_X1.unsqueeze(dim=-3)
        product = torch.einsum('...np,...mdp->...mnd', X1, X2) - diag_X1 - diag_X2
        result = torch.einsum('bhnmd,bmd->bhnd', result, random_sign)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)

    return result

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
    
    
def kernel_RS_RBF0(X1, X2=None, X2_accu=False, random_sign=None):

    # todo

    if X2 is None:
        X2 = X1
        X2_accu = False

    diag_X1 = torch.abs(X1) ** power
    diag_X1 = torch.sum(diag_X1, dim=-1) / scale
    diag_X1 = diag_X1.unsqueeze(dim=-1)
    diag_X2 = torch.abs(X2) ** power
    diag_X2 = torch.sum(diag_X2, dim=-1) / scale
    diag_X2 = diag_X2.unsqueeze(dim=-2)

    if X2_accu:
        diag_X1 = diag_X1.unsqueeze(dim=-3)
        product = torch.einsum('...np,...mdp->...mnd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)
        result = torch.transpose(result, 2, 3) # nmd
        result = result * random_sign
        result = result.sum(dim=3)
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

def uniform_sketching(n, nb_rows, nb_columns, device):
    w = torch.ones(n, device=device)
    S = torch.multinomial(w, nb_rows * nb_columns, replacement=False).reshape(nb_rows, nb_columns)
    random_sign = (torch.randint(2, S.shape, device=device) * 2 - 1) * ( math.sqrt(n) / nb_rows / nb_columns)
    return S, random_sign

# not used
def pinv(X, hermitian = True, eps = 1e-4):
    if hermitian:
        Sigma, U = torch.symeig(X, eigenvectors=True)
        Sigma = torch.where(Sigma > eps, 1 / Sigma, torch.tensor(0.))
        # print(U.shape, Sigma.shape)
        res = torch.einsum('...md,...nd,...d->...mn', U, U, Sigma)
        return res
    else:
        return torch.pinverse(X)

# Now CG_solve_lin works for A X = B, where A is spare, B is dense.
def CG_solve_lin(A, B = "I", n_iter = 6, tol = 1e-14):

    # Solve X from linear system AX = B, default B = I
    # stop criterion:
    #  reach n_iter, or eps = mean(R_ij^2) < tol

    # A is ...* m * m Tensor
    # B is ... * m Tensor or ... * m * n tensor
    A_dim   = len(A.size())
    A_shape = [A.size(i) for i in range(A_dim)]

    # create identity tensor (last 2 dims is I, repeat for other dims)
    if B == "I":    
        shape_ls = A_shape[ :-2] + [1, 1]
        B = torch.eye(A.size(-1), device = A.device).repeat(shape_ls)

    # make sure the dim(B) = dim(A) or dim(A) - 1
    if len(B.size()) == len(A.size()):
        pass
    elif len(B.size()) == (len(A.size()) -1):
        # reshape B to dim(A)
        B0_shape = [B.size(i) for i in range(len(B.size()))]
        if B0_shape[: -1] == A_shape[: -2]:
            B_shape = B0_shape + [1]
            B = B.reshape(B_shape)
        else:
            print("the first dims of B must match these of mat")
            return 0
    else:
        print("dim(B) should equal to dim(mat) or dim(mat) - 1!")
        return 0

    # X = torch.zeros(B.size())
    X = 0
    D, R0 = B, B
    i, eps = 0, 1e-3

    while i < n_iter and eps > tol:
        # r_{i-1}'  r_{i-1}
        # num = torch.einsum('...ij, ...ij -> ...j', R0, R0)
        num = (R0 * R0).sum(-2)
        # num[num<eps] = eps
        # d_{i-1}' A d_{i-1}
        # den = torch.einsum('...ji, ...jk, ...ki -> ...i', D, A, D)

        AD = A @ D
        # den = torch.sum(D * AD, -2) + eps
        den = torch.sum(D * AD, -2)
        # den[den<eps] = eps

        # alpha = (r_{i-1}'  r_{i-1}) / (d_{i-1}' A d_{i-1})
        alpha = num / den
        # X_i = X_{i-1} + alpha d_{i-1}
        X = X + torch.einsum('...j, ...ij -> ...ij', alpha, D)
        # r_i = r_{i-1} - alhpa A d_{i-1}
        # Ad = torch.einsum('...ij, ...jk -> ...ik', A, D)

        # print()
        R = R0 - torch.einsum('...j, ...ij -> ...ij', alpha, AD)   
        # beta = (r_{i}' r_{i}) / (r_{i-1}' r_{i-1})
        # beta = torch.einsum('...ij, ...ij -> ...j', R, R) / num
        # num = num + eps
        # num[num<eps] += eps
        
        
        beta = (R * R).sum(-2) / num
        # d_i = r_u + beta d_{i-1}
        D = R + torch.einsum('...j, ...ij -> ...ij', beta, D)
        R0 = R
        # inf norm: max(sum(abs(x), dim=1))
        # do row abs sum frst. Then take max
        # eps = torch.mean(R0 * R0)
        # eps = 1
        i += 1
        # print('eps: ', eps, "iter: ", i)
        # print("iter: ", i, 'X[16]inCG', X[16])

    return X
    
    
def CG_solve_lin_check(A, B = "I", n_iter = 6, tol = 1e-14):

    # Solve X from linear system AX = B, default B = I
    # stop criterion:
    #  reach n_iter, or eps = mean(R_ij^2) < tol

    # A is ...* m * m Tensor
    # B is ... * m Tensor or ... * m * n tensor
    A_dim   = len(A.size())
    A_shape = [A.size(i) for i in range(A_dim)]

    # create identity tensor (last 2 dims is I, repeat for other dims)
    if B == "I":    
        shape_ls = A_shape[ :-2] + [1, 1]
        B = torch.eye(A.size(-1), device = A.device).repeat(shape_ls)

    # make sure the dim(B) = dim(A) or dim(A) - 1
    if len(B.size()) == len(A.size()):
        pass
    elif len(B.size()) == (len(A.size()) -1):
        # reshape B to dim(A)
        B0_shape = [B.size(i) for i in range(len(B.size()))]
        if B0_shape[: -1] == A_shape[: -2]:
            B_shape = B0_shape + [1]
            B = B.reshape(B_shape)
        else:
            print("the first dims of B must match these of mat")
            return 0
    else:
        print("dim(B) should equal to dim(mat) or dim(mat) - 1!")
        return 0

    # X = torch.zeros(B.size())
    X = 0
    D, R0 = B, B
    i, eps = 0, 1e-3

    while i < n_iter and eps > tol:
        # r_{i-1}'  r_{i-1}
        # num = torch.einsum('...ij, ...ij -> ...j', R0, R0)
        num = (R0 * R0).sum(-2)
        # num[num<eps] = eps
        # d_{i-1}' A d_{i-1}
        # den = torch.einsum('...ji, ...jk, ...ki -> ...i', D, A, D)

        AD = A @ D
        # den = torch.sum(D * AD, -2) + eps
        den = torch.sum(D * AD, -2)
        # den[den<eps] = eps

        # alpha = (r_{i-1}'  r_{i-1}) / (d_{i-1}' A d_{i-1})
        alpha = num / den
        # X_i = X_{i-1} + alpha d_{i-1}
        X = X + torch.einsum('...j, ...ij -> ...ij', alpha, D)
        # r_i = r_{i-1} - alhpa A d_{i-1}
        # Ad = torch.einsum('...ij, ...jk -> ...ik', A, D)

        # print()
        R = R0 - torch.einsum('...j, ...ij -> ...ij', alpha, AD)   
        # beta = (r_{i}' r_{i}) / (r_{i-1}' r_{i-1})
        # beta = torch.einsum('...ij, ...ij -> ...j', R, R) / num
        # num = num + eps
        # num[num<eps] += eps
        
        
        beta = (R * R).sum(-2) / num
        # d_i = r_u + beta d_{i-1}
        D = R + torch.einsum('...j, ...ij -> ...ij', beta, D)
        R0 = R
        # inf norm: max(sum(abs(x), dim=1))
        # do row abs sum frst. Then take max
        # eps = torch.mean(R0 * R0)
        # eps = 1
        i += 1
        # print('eps: ', eps, "iter: ", i)
        # print("iter: ", i, 'X[16]inCG', X[16])
        print(X.abs().max())
        
    return X
        
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

 

class SketchedAttentionRBF(nn.Module):

    def __init__(self, config):
        """
        nb_features

        """
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

        # self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling, qr_uniform_q = qr_uniform_q)
        # self.create_sketching = partial(uniform_sketching, n=2*n, nb_rows = accumulation, nb_columns = nb_features, device=self.device)
        # sketching_matrix, random_sign = self.create_sketching()
        # self.register_buffer('sketching_matrix', sketching_matrix)
        # self.register_buffer('random_sign', random_sign)

        # TODO: enable other kernel function?
        # self.kernel_fn = kernel_fn
        if config["sketched_kernel"] == "kernel_RS_RBF":
            self.kernel_fn = kernel_RS_RBF

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        # self.no_projection = no_projection
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
        # print('S', S[16])
        # print(torch.unique(S[16], return_counts=True))
        return S, random_sign

    def forward(self, q, k, v, mask):

        
        device = q.device
        b, h, n, d = q.shape
        
        # data_normalizer = n**(1/(2*d+3))
        data_normalizer = (32 ** -0.25)
        

        q = q * (mask[:, None, :, None] * data_normalizer)
        k = k * (mask[:, None, :, None] * data_normalizer)
        v = v * mask[:, None, :, None]
        
        # print(q[0, 0], v[0, 0])
        
        non_padding_num = mask.sum(-1) # b
        # print(non_padding_num)

        self.sketching_matrix, self.random_sign = self.uniform_sketching(
            n, self.accumulation, self.nb_features, non_padding_num) # bmd
        
        # print(self.sketching_matrix, self.random_sign.shape)
        
        create_kernel_sketch = partial(kernel_sketch, kernel_fn = self.kernel_fn,
           sketching_matrix = self.sketching_matrix, random_sign=self.random_sign, device = device)
        AS = create_kernel_sketch(q, k)  # b,h,2n, nb_feat
        Q = AS[:,:,:n] # b, h, n, nb_feat
        
        # print(K[0, 2])
        # STAS = AS[:,:,self.sketching_matrix].permute(0,1,4,2,3) * self.random_sign # bhmdp -> bhpmd
        # STAS = torch.transpose(STAS, 2, 4).sum(dim=3) # bhpmd -> bhdmp -> bhdp
        
        STAS = AS.transpose(1, 2)[torch.arange(b)[:, None, None], self.sketching_matrix] # bnhd -> bmdhd
        STAS = torch.einsum('bmdhe,bmd->bhde', STAS, self.random_sign) # bmdhd -> bhdd
        # STAS = (STAS.permute(3,4,0,1,2) * self.random_sign).sum(-2).permute(2,0,3,1) # hdbmd -> hdbd -> bhdd
        # STAS = STAS + 1e-2*torch.eye(STAS.shape[-1], device=self.device)
        
        # print(AS.shape, STAS[0, 0])
        
        STAS = STAS + 1e-1*torch.eye(STAS.shape[-1], device=self.device)
        # K = K + torch.eye(K.shape[-1], device=self.device)
        # print(torch.linalg.svd(STAS[0, 0])[1])
        # print(STAS[0, 0])
        # K = la.solve(K + 1e-4*torch.eye(K.shape[-1])
        # K = la.solve(STAS, torch.transpose(AS[:,:,n:], 2, 3))
        # K = CG_solve_lin(STAS, torch.transpose(AS[:,:,n:], 2, 3), 5)
        # K = torch.transpose(K, 2, 3) * mask[:, None, :, None]
        
        K = AS[:,:,n:]
        
        
        ##################################################################
        D_STAS_inv = 1 / STAS.sum(-1)
        # D_STAS_inv = D_STAS_inv * D_STAS_inv
        D_STAS_inv = torch.sqrt( D_STAS_inv)
        STAS = torch.einsum("...d,...de,...e->...de", D_STAS_inv, STAS, D_STAS_inv)
        
        
        
        STAS_inv = iterative_inv(STAS, 6)
        
        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv) @ STAS_inv
        
        # print(K[0, 0])
        
        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv) * mask[:, None, :, None]
        
        ##################################################################

        context = torch.einsum('...nd,...ne->...de', K, v)
        out = torch.matmul(Q, context)    
        
        return out   

        

       


        
        
scale = 2 
power = 2

