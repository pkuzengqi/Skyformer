import sys
# sys.path.append("../data/long-range-arena-main/lra_benchmarks/text_classification/")
import lra_text_input_pipeline as input_pipeline
import numpy as np
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# note: nystromformer code is doing textlength=1000+24
train_ds, eval_ds, test_ds, encoder = input_pipeline.get_tc_datasets(
    n_devices = 1, task_name = "imdb_reviews", data_dir = None,
    batch_size = 1, fixed_vocab = None, max_length = 4000)

mapping = {"train":train_ds, "dev": eval_ds, "test":test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0":np.concatenate([inst["inputs"].numpy()[0], np.zeros(96, dtype = np.int32)]),
            "label":inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")
    with open(f"../data/lra_processed/lra-text.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
