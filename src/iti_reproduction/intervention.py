from pathlib import Path

import numpy as np
import pandas as pd
import pyvene as pv
import torch
from datasets import load_dataset
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

from iti_reproduction.interveners import ITI_Intervener, wrapper
from iti_reproduction.scheme import ITIConfig

# Specific pyvene imports
from iti_reproduction.utils import (
    alt_tqa_evaluate,
    get_com_directions,
    get_separated_activations,
    get_top_heads,
    layer_head_to_flattened_idx,
)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def intervene(cfg: ITIConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    df = pd.read_csv("./TruthfulQA.csv")
    dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
    golden_q_order = list(dataset["question"])
    df = df.sort_values(
        by="Question", key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)})
    )  # get two folds using numpy
    if cfg.debug:
        df = df.sample(n=10, random_state=cfg.seed).reset_index(drop=True)
    fold_idxs = np.array_split(np.arange(len(df)), cfg.num_fold)

    # create model
    model_name_or_path = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    # load activations
    base_dir = Path("./features/")
    head_wise_activations = np.load(
        base_dir
        / f"{cfg.model_name}_{cfg.dataset_name}_head_wise.npy".replace("/", "_")
    )
    labels = np.load(
        base_dir / f"{cfg.model_name}_{cfg.dataset_name}_labels.npy".replace("/", "_")
    )
    # reshape to (batch, seq_len, num_heads, head_dim)
    head_wise_activations = rearrange(
        head_wise_activations, "b l (h d) -> b l h d", h=num_heads
    )
    # tuning dataset: no labels used, just to get std of activations along the direction
    activations_dataset = (
        cfg.dataset_name if cfg.activation_dataset is None else cfg.activation_dataset
    )
    tuning_activations = np.load(
        base_dir
        / f"{cfg.model_name}_{activations_dataset}_head_wise.npy".replace("/", "_")
    )
    tuning_activations = rearrange(
        tuning_activations, "b l (h d) -> b l h d", h=num_heads
    )
    tuning_labels = np.load(
        base_dir
        / f"{cfg.model_name}_{activations_dataset}_labels.npy".replace("/", "_")
    )

    separated_head_wise_activations, separated_labels, idxs_to_split_at = (
        get_separated_activations(labels, head_wise_activations)
    )
    # run k-fold cross validation
    results = []
    for i in range(cfg.num_fold):
        train_idxs = np.concatenate(
            [fold_idxs[j] for j in range(cfg.num_fold) if j != i]
        )
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(
            train_idxs, size=int(len(train_idxs) * (1 - cfg.val_ratio)), replace=False
        )
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        Path("splits/").mkdir(parents=True, exist_ok=True)
        df.iloc[train_set_idxs].to_csv(
            f"splits/fold_{i}_train_seed_{cfg.seed}.csv", index=False
        )
        df.iloc[val_set_idxs].to_csv(
            f"splits/fold_{i}_val_seed_{cfg.seed}.csv", index=False
        )
        df.iloc[test_idxs].to_csv(
            f"splits/fold_{i}_test_seed_{cfg.seed}.csv", index=False
        )

        # get directions
        # if cfg.use_center_of_mass:
        # 解答がTrueとFalseのサンプルにおける方向ベクトルの平均の差の計算をしている
        com_directions = get_com_directions(
            num_layers,
            num_heads,
            train_set_idxs,
            val_set_idxs,
            separated_head_wise_activations,
            separated_labels,
        )
        # else:
        #     com_directions = None
        # top_heads: List[Tuple[int, int]] = [(layer, head), ...]
        top_heads, probes = get_top_heads(
            train_set_idxs,
            val_set_idxs,
            separated_head_wise_activations,
            separated_labels,
            num_layers,
            num_heads,
            cfg.seed,
            2,
            cfg.use_random_dir,
        )

        print("Heads intervened: ", sorted(top_heads))

        interveners = []
        pv_config = []
        top_heads_by_layer = {}
        for (
            layer,
            head,
        ) in top_heads:
            if layer not in top_heads_by_layer:
                top_heads_by_layer[layer] = []
            top_heads_by_layer[layer].append(head)
        for layer, heads in top_heads_by_layer.items():
            direction = torch.zeros(head_dim * num_heads).to(device)
            for head in heads:
                # direction
                dir = torch.tensor(
                    com_directions[layer_head_to_flattened_idx(layer, head, num_heads)],
                    dtype=torch.float32,
                ).to(device)
                # 正規化
                dir = dir / torch.norm(dir)
                activations = torch.tensor(
                    tuning_activations[:, layer, head, :], dtype=torch.float32
                ).to(device)  # batch x 128
                # ここは標準偏差を計算している？
                # 論文と付き合わせながら確認する必要あり
                proj_vals = activations @ dir.T
                proj_val_std = torch.std(proj_vals)
                direction[head * head_dim : (head + 1) * head_dim] = dir * proj_val_std
            intervener = ITI_Intervener(
                direction, cfg.alpha
            )  # head=-1 to collect all head activations, multiplier doens't matter
            interveners.append(intervener)
            pv_config.append(
                {
                    "component": f"model.layers[{layer}].self_attn.o_proj.input",
                    "intervention": wrapper(intervener),
                }
            )
        intervened_model = pv.IntervenableModel(pv_config, model)

        filename = f"{cfg.model_prefix}{cfg.model_name}_seed_{cfg.seed}_top_{num_heads}_heads_alpha_{int(cfg.alpha)}_fold_{i}"

        if cfg.use_center_of_mass:
            filename += "_com"
        if cfg.use_random_dir:
            filename += "_random"
        Path("./results_dump/answer_dump/").mkdir(parents=True, exist_ok=True)
        Path("./results_dump/summary_dump/").mkdir(parents=True, exist_ok=True)

        curr_fold_results = alt_tqa_evaluate(
            models={cfg.model_name: intervened_model},
            metric_names=["mc", "bleu", "rouge", "bleurt"],
            # metric_names=["bleurt"],
            input_path=f"splits/fold_{i}_test_seed_{cfg.seed}.csv",
            output_path=f"./results_dump/answer_dump/{filename}.csv",
            summary_path=f"./results_dump/summary_dump/{filename}.csv",
            device=device,
            interventions=None,
            intervention_fn=None,
            instruction_prompt=cfg.instruction_prompt,
            judge_name=cfg.judge_name,
            info_name=cfg.info_name,
            separate_kl_device=device,
            orig_model=model,
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)

    results = np.array(results)
    final = results.mean(axis=0)

    print(
        f"alpha: {cfg.alpha}, heads: {num_heads}, True*Info Score: {final[1] * final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}"
    )
