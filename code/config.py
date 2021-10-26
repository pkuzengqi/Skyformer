

# seq_len=4096, nb_feat=256
# seq_len=512, nb_feat=64
### big bird: n*nbf = (r+w+g)*b*n, block_size = nb_feat / 10

listops = {
              "dataset":{
                  "train":96000,
                  "dev":2000,
                  "test":2000,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":64, #64
                  "transformer_dim":64, #64
                  "transformer_hidden_dim":128, #128
                  "head_dim":32, #32
                  "num_head":2, #2
                  "num_layers":2,
                  "vocab_size":32,
                  "max_seq_len":2000,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes":10,
              },
              "training":{
                  "batch_size":32, # large 256, paper 32
                  "learning_rate":0.0001,
                  "warmup":1000,
                  "lr_decay":"linear",
                  "weight_decay":0,
                  "eval_frequency":500, #50
                  "num_train_steps":50000,#5000/2=2500
                  "num_init_steps":1000,
                  "num_eval_steps":62,
                  "patience":10, #10
              },
              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":1},
 

                  "sketchedRNRBF1128":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRNRBF1256":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":256, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRNRBF":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":64, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRNRBF32":{"bz_rate":1, "nb_features":64, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRBF":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":64, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRBF1128":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRBF1256":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":256, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRBF32":{"bz_rate":1, "nb_features":64, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedRBF32256":{"bz_rate":1, "nb_features":256, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},

                  "sketchedKDE1128":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedKDE1256":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":256, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedKDE32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedKDE32256":{"bz_rate":1, "nb_features":256, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},

                  "sketched":{"bz_rate":1,"nb_features":64, "sketched_kernel":"kernel_RS_SM", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketched32":{"bz_rate":1,"nb_features":64, "sketched_kernel":"kernel_RS_SM", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  "sketchedQKV":{"bz_rate":1, "Q_dim":1,"K_dim":1,"V_dim":32, "nb_features":64, "sketched_kernel":"kernel_RS_SM", "accumulation":1, "sampling_factor":4, "no_projection":False},

                  "nystrom":{"bz_rate":1,"num_landmarks":128},
                  "linformer":{"bz_rate":1,"linformer_k":128},
                  "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
                  "informer":{"bz_rate":1,"in_nb_features":128},
                  "reformer":{"bz_rate":1,"num_hash":2},
                  "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},

                  }
          }
pathfinder = {
           "model":{
               "learn_pos_emb":True,
               "tied_weights":False,
               "embedding_dim":64, #64
               "transformer_dim":64, #64
               "transformer_hidden_dim":128, #128
               "head_dim":32, #32
               "num_head":2, #2
               "num_layers":2,
               "vocab_size":512,
               "max_seq_len":1024,
               "dropout_prob":0.1,
               "attention_dropout":0.1,
               "pooling_mode":"MEAN",
               "num_classes": 2,
           },
           "training":{
               "batch_size":128, #large 512,small128, paper128
               "learning_rate":0.0002,
               "warmup":312, #312
               "lr_decay":"linear",
               "weight_decay":0,
               "eval_frequency":500, #312
               "num_train_steps":50000, #paper62400
               "num_init_steps":3500,
               "num_eval_steps":156, #312
               "patience":10, #10
           },
           "extra_attn_config":{
               "softmax":{"bz_rate":1,},
               "softmaxRBF32":{"bz_rate":1},
               "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},

               "nystrom":{"bz_rate":1,"num_landmarks":128},
               "linformer":{"bz_rate":1,"linformer_k":128},
               "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
               "informer":{"bz_rate":2,"in_nb_features":128},
               "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
               "reformer":{"bz_rate":1,"num_hash":2},

           }
       }
retrieval={
              "dataset":{
                  "train":147086,
                  "dev":18090,
                  "test":17437,
              },
              "model":{
                  "learn_pos_emb":True,
                  "tied_weights":False,
                  "embedding_dim":64, #64
                  "transformer_dim":64, #64
                  "transformer_hidden_dim":128, #128
                  "head_dim":32, #32
                  "num_head":2, #2
                  "num_layers":2,
                  "vocab_size":512,
                  "max_seq_len":4000,
                  "dropout_prob":0.1,
                  "attention_dropout":0.1,
                  "pooling_mode":"MEAN",
                  "num_classes": 2,
              },
              "training":{
                  "batch_size":16, # large64, paper32, past16
                  "learning_rate":0.0002,
                  "warmup":800,
                  "lr_decay":"linear",
                  "weight_decay":0,
                  "eval_frequency":1000, #300
                  "num_train_steps":50000, #30000/4=7500
                  "num_init_steps":3000,
                  "num_eval_steps":300, #565
                  "patience":10, #10
              },
              "extra_attn_config":{
                  "softmax":{"bz_rate":1,},
                  "softmaxRBF32":{"bz_rate":2},
                  "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
                  
                  "nystrom":{"bz_rate":1,"num_landmarks":128},
                  "linformer":{"bz_rate":1,"linformer_k":128},
                  "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
                  "informer":{"bz_rate":1,"in_nb_features":128},
                  "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
                  "reformer":{"bz_rate":1,"num_hash":2},
              }
          }
text={
         "dataset":{
             "train":25000,
             "dev":25000,
             "test":25000,
         },
         "model":{
             "learn_pos_emb":True,
             "tied_weights":False,
             "embedding_dim":64, #64
             "transformer_dim":64, #64
             "transformer_hidden_dim":128, #128
             "head_dim":32, #32
             "num_head":2, #2
             "num_layers":2,
             "vocab_size":512,
             "max_seq_len":4000, # will be 4096
             "dropout_prob":0.1,
             "attention_dropout":0.1,
             "pooling_mode":"MEAN",
             "num_classes": 2,
         },
         "training":{
             "batch_size":16, #small16, large128, paper32
             "learning_rate":0.0001,
             "warmup":80, # 80
             "lr_decay":"linear",
             "weight_decay":0,
             "eval_frequency":500, #500
             "num_train_steps":50000, # 20000
             "num_init_steps":3000,
             "num_eval_steps":200, #781
             "patience":10, #10
         },
         "extra_attn_config":{
             "softmax":{"bz_rate":1},
             "softmaxRBF32":{"bz_rate":2},
             "sketchedRBF32128":{"bz_rate":1, "nb_features":128, "sketched_kernel":"kernel_RS_RBF", "accumulation":1, "sampling_factor":4, "no_projection":False},
             
             "nystrom":{"bz_rate":1,"num_landmarks":128},
             "linformer":{"bz_rate":1,"linformer_k":128},
             "performer":{"bz_rate":1,"rp_dim":128, "kernel_type":"exp"}, #rp_dim = nb_features
             "informer":{"bz_rate":1,"in_nb_features":128},
             "bigbird":{"bz_rate":1,"num_random_blocks":3, "block_size":64},
             "reformer":{"bz_rate":1,"num_hash":2},
         }
     }

Config = {
    "lra-listops":listops,
    "lra-pathfinder":pathfinder,
    "lra-retrieval":retrieval,
    "lra-text":text,
}

Config["lra-pathfinder32-curv_contour_length_14"] = Config["lra-pathfinder"]

