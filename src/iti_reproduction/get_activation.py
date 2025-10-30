import pickle
from pathlib import Path

import numpy as np
import pyvene as pv
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from iti_reproduction.interveners import Collector, wrapper
from iti_reproduction.scheme import ITIConfig
from iti_reproduction.utils import (get_llama_activations_pyvene,
                                    tokenized_tqa, tokenized_tqa_gen,
                                    tokenized_tqa_gen_end_q)


def get_activation(cfg: ITIConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map=cfg.device)
    match cfg.dataset_name:
        case "tqa_mc2":
            dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")[
                "validation"
            ]
            formatter = tokenized_tqa
        case "tqa_gen":
            dataset = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
            formatter = tokenized_tqa_gen
        case "tqa_gen_end_q":
            dataset = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
            formatter = tokenized_tqa_gen_end_q
        case _:
            raise ValueError("Invalid dataset name")

    if cfg.debug:
        dataset = dataset.select([0])

    print("Tokenizing prompts")
    if cfg.dataset_name == "tqa_gen" or cfg.dataset_name == "tqa_gen_end_q":
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(
            f"./features/{cfg.model_name}_{cfg.dataset_name}_categories.pkl".replace(
                "/", "_"
            ),
            "wb",
        ) as f:
            pickle.dump(categories, f)
    else:
        prompts, labels = formatter(dataset, tokenizer)
    collectors = []
    pv_config = []

    for layer in range(model.config.num_hidden_layers):
        collector = Collector(
            multiplier=0, head=-1
        )  # head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append(
            {
                "component": f"model.layers[{layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            }
        )
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(
            collected_model, collectors, prompt, cfg.device
        )
        # 一番最後の特徴量を持ってくる
        all_layer_wise_activations.append(layer_wise_activations[:, -1, :].copy())
        all_head_wise_activations.append(head_wise_activations.copy())

    print("Saving labels")
    base_dir = Path("./features")
    base_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        base_dir / f"{cfg.model_name}_{cfg.dataset_name}_labels.npy".replace("/", "_"),
        labels,
    )

    print("Saving layer wise activations")
    np.save(
        base_dir
        / f"{cfg.model_name}_{cfg.dataset_name}_layer_wise.npy".replace("/", "_"),
        all_layer_wise_activations,
    )

    print("Saving head wise activations")
    np.save(
        f"./features/{cfg.model_name}_{cfg.dataset_name}_head_wise.npy".replace(
            "/", "_"
        ),
        all_head_wise_activations,
    )
