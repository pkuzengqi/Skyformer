import os
import sys
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import redirect_stdout
from config import Config
from models.model_LRA import ModelForSC, ModelForSCDual
from models.dataset_LRA import LRADataset






def print_summary(summary, save_if_improved, model, checkpoint_path):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])


    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]

    summary_round = {}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []

def step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
             accumu_steps, init_t, summary, component, step_idx, writer=None):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            # with torch.cuda.amp.autocast():
            partial_outputs = model(**partial_inputs)

            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]


            amp_scaler.scale(partial_outputs["loss"]).backward() # loss.backward()

        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        amp_scaler.unscale_(optimizer)


        # for name, parms in model.named_parameters():
        #     print('BEFORE -->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
        # for name, parms in model.named_parameters():
        #     print('AFTER-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)


        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    if step_idx%100==0:
        print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

    if writer is not None:
        writer.add_scalar('loss', loss, step_idx)
        writer.add_scalar('accu', accu, step_idx)

    return outputs

def train_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
              training_config, summary, writer):

    accumu_steps = training_config['accumu_steps']
    checkpoint_path = training_config['checkpoint_path']
    # best_dev_loss = float(1e10)
    best_dev_accu = 0
    total_step = training_config["num_train_steps"]

    init_t = time.time()

    model.train()
    for train_step_idx in range(total_step):
        outputs = step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
                           accumu_steps, init_t, summary, component='train', step_idx=train_step_idx,writer=writer)

        if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
            print_summary(summary["train"], False, model, checkpoint_path)
            model.eval()
            for dev_step_idx in range(training_config["num_eval_steps"]):
                outputs = step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
                                   accumu_steps, init_t, summary, component='dev', step_idx=dev_step_idx)
            # dev_loss = np.mean(summary["dev"]["loss"])
            # if  dev_loss < best_dev_loss:
            #     best_dev_loss = dev_loss
            dev_accu = np.mean(summary["dev"]["accu"])
            if dev_accu > best_dev_accu:
                best_dev_accu = dev_accu
                if (train_step_idx + 1) > total_step * 0.2:
                    torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
                    print('best model saved: step = ',train_step_idx, 'dev accu = ',dev_accu)

                    # nearest5k = str(math.ceil((train_step_idx+1)/5000))
                    # torch.save({"model_state_dict":model.state_dict()}, checkpoint_path+nearest5k)
                    # print('best model saved to ',nearest5k)

            print_summary(summary["dev"], True, model, checkpoint_path)
            model.train()





    print('total training step (k): {}'.format(total_step/1000.0))
    print("total training time (s): {}".format(int(time.time()-init_t)))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))


def eval_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
             training_config, summary):
    accumu_steps = training_config['accumu_steps']
    checkpoint_path = training_config['checkpoint_path']
    init_t = time.time()
    model.eval()
    try:
        for test_step_idx in itertools.count():
            outputs = step_LRA(model, optimizer, lr_scheduler, ds_iter,amp_scaler,
                               accumu_steps, init_t, summary, component='test', step_idx=test_step_idx)
    except StopIteration:
        print_summary(summary["test"], False, model, checkpoint_path)

def get_args():
    ### parse arguments###
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default="train",
                        help="train eval")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument("--attn", type = str, default="softmaxQKV",
                        help = "softmax, nystrom, linformer, informer, performer, bigbird, sketched, skeinb,skein, skein0, skeini")
    parser.add_argument("--task", type = str, default="lra-listops",
                        help = "lra-listops, lra-retrieval, lra-text, lra-pathfinder32-curv_contour_length_14")
    parser.add_argument('--random', type=int, default=42)
    parser.add_argument("--do_debug", type=int, default=1, help='set to 1 if in debug mode; otherwise set to 0 ')
    parser.add_argument("--bz", type=int, default=0, help='set to 0 if using values in config')
    parser.add_argument("--lr", type=float, default=0, help='set to 0 if using values in config')
    parser.add_argument("--init", type=int, default=0, help='set to 0 if using values in config')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if args.task == 'lra-pathfinder':
        args.task = 'lra-pathfinder32-curv_contour_length_14'


    ## use 0 to debug; set args.do_debug to False when run in shell
    if args.do_debug > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('os.environ["CUDA_VISIBLE_DEVICES"]: \t', end='')
    print(os.environ["CUDA_VISIBLE_DEVICES"])



    ### get model config ###
    model_config = Config[args.task]["model"]
    if args.attn in Config[args.task]["extra_attn_config"]:
        model_config.update(Config[args.task]["extra_attn_config"][args.attn])
    model_config["mixed_precision"] = True
    model_config["attn_type"] = args.attn
    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    model_config["random_seed"] = args.random

    training_config = Config[args.task]["training"]
    if args.bz > 0:
        training_config["batch_size"] = args.bz
    if args.lr > 0:
        training_config["learning_rate"] = args.lr
    if args.init > 0:
        training_config["num_init_steps"] = args.init
    if args.do_debug > 0:
        training_config["num_train_steps"] = 20
        training_config["eval_frequency"] = 2
        training_config["warmup"] = 2

        ### log preparation ###
    log_dir = './log-same-step-{}/'.format(args.random)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.task)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.checkpoint))
    redirect_stdout(open(log_path, 'w'))
    summary = {
        component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
        for component in ["train", "dev", "test"]
    }
    writer = SummaryWriter(os.path.join(log_dir,'{}.tensorboard'.format(args.checkpoint)))

    print(json.dumps([model_config, training_config], indent = 4))


    ###  set the random seeds for deterministic results. ####
    SEED = args.random
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True





    ### model preparation ###
    if args.task == "lra-retrieval":
        model = ModelForSCDual(model_config)
    else:
        model = ModelForSC(model_config)

    # mark: specify the checkpoint to load
    # checkpoint_path = os.path.join('./checkpoints_lra20k', 'lra-pathfinder32-curv_contour_length_14_softmaxQKV.47.model_15000')


    # checkpoint_dir = './checkpoints_lra_samestep-{}'.format(args.random)
    # if not os.path.exists(checkpoint_dir):
    #     os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join('./checkpoints_lra20k', '{}.{}.model'.format(args.checkpoint, args.random))
    training_config["checkpoint_path"] = checkpoint_path
    # if os.path.exists(checkpoint_path) and checkpoint_path != './checkpoints/test.model':
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     print("model loaded from: " + checkpoint_path)

    model = model.cuda()
    print(model)
    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)


    device_ids = list(range(torch.cuda.device_count()))
    # print(f"GPU list: {device_ids}")
    # model = nn.DataParallel(model, device_ids = device_ids)



    ### data preparation ###
    if args.do_debug > 0:
        a=enumerate(DataLoader(LRADataset(f"./data/lra_processed/{args.task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True))
        ds_iter = {
            "train":a,
            "dev":a,
            "test":a,
        }
    else:
        ds_iter = {
            "train":enumerate(DataLoader(LRADataset(f"./data/lra_processed/{args.task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
            "dev":enumerate(DataLoader(LRADataset(f"./data/lra_processed/{args.task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
            "test":enumerate(DataLoader(LRADataset(f"./data/lra_processed/{args.task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True)),
        }

    ### training preparation ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"],
        betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer = optimizer,
        max_lr = training_config["learning_rate"],
        pct_start = training_config["warmup"] / training_config["num_train_steps"],
        anneal_strategy = training_config["lr_decay"],
        total_steps = training_config["num_train_steps"]
    )

    amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None


    # accumu_steps = max(training_config["batch_size"] // len(device_ids) // model_config["gpu_memory"], 1)
    accumu_steps = model_config["bz_rate"] if "bz_rate" in model_config else 1
    # accumu_steps = 1
    print(f"accumu_steps={accumu_steps}")
    training_config['accumu_steps'] = accumu_steps



    ### train ###
    if args.mode == 'train':
        train_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
                  training_config, summary, writer)

    ### eval ###
    if os.path.exists(checkpoint_path) and checkpoint_path != './checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
    eval_LRA(model, optimizer, lr_scheduler, ds_iter, amp_scaler,
             training_config, summary)



if __name__ == '__main__':
    main()
