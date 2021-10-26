import time
import math
import torch
import torch.utils.checkpoint
from torch import nn
from self_attention import BertConfig, BertSelfAttention
from sketched_attention import SketchedAttention
from performer_attention import FastAttention as PerformerAttention





# archive
class AttentionWrapper(nn.Module):
    '''
    transform sketched_attention to BertSelfAttention style

    BertSelfAttention style
    input: hidden_states = torch.randn(1, 512, 768)
    output: tuple ([bz=1,seq_len=512,dim=768=64*12], )

    SketchAttention style
    # queries / keys / values with heads already split and transposed to first dimension
    # 8 heads, dimension of head is 64, sequence length of 512
    q = torch.randn(1, 8, 512, 64)
    k = torch.randn(1, 8, 512, 64)
    v = torch.randn(1, 8, 512, 64)
    out = attn_fn(q, k, v) # (1, 8, 512, 64) 

    :return:
    '''
    def __init__(self, config, attn_cls):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        # call sketch_attention
        # __init__(self, n=512, dim_heads=64, kernel_fn = kernel_SM, nb_features = None, accumulation = 7, no_projection = False):
        self.self = attn_cls(n=config.max_position_embeddings,
                             dim_heads=config.hidden_size / config.num_attention_heads)

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        # [1,7,768] --> [1,7,12,64] --> [bz=1,num_heads=12,seq_len=7,head_sim=64]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # input: hidden_states[1, 7, 768]
        mixed_query_layer = self.query(hidden_states) # Linear(768,768)

        # [1,7,768] --> [1,7,12,64] --> [bz=1,num_heads=12,seq_len=7,head_sim=64]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if True in torch.isinf(query_layer):
            print()
        if True in torch.isinf(key_layer):
            print()
        if True in torch.isinf(value_layer):
            print()

        # q_layer [bz = 1, num_heads = 12, seq_len = 7, head_sim = 64]
        # sketch_attention input q = torch.randn(bz=1, num_heads=12, seq_len=512, head_dim=64)
        context_layer = self.self(query_layer,key_layer,value_layer)
        # sketch_attention output (1, 8, 512, 64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # dim(1,512,12,64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #shape(1,512,768)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
        # outputs tuple ([bz=1,seq_len=512,dim=768], attention)
        return outputs



if __name__ == "__main__":

    config = BertConfig()
    bz = 16
    seqlen = 256
    hiddim = 768
    config.hidden_size = hiddim
    config.max_position_embeddings = seqlen


    def test_inference(attn_fn):
        n = 3
        hidden_states = torch.randn(bz, seqlen, hiddim)
        st = time.time()
        for i in range(n):
            out = attn_fn(hidden_states)
            # print(out[0].shape)
            # outputs tuple ([bz=1,seq_len=512,dim=768], )
        en = time.time()
        print('%.4f'%((en - st) / n))

    print('SketchedAttention')
    attn_fn = AttentionWrapper(config, SketchedAttention)
    test_inference(attn_fn)

    print('PerformerAttention')
    attn_fn = AttentionWrapper(config, PerformerAttention)
    test_inference(attn_fn)

    print('BertSelfAttention')
    attn_fn = BertSelfAttention(config)
    test_inference(attn_fn)


