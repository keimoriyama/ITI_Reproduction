import pickle

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from iti_reproduction.scheme import ActivationConfig
from iti_reproduction.utils import (tokenized_tqa, tokenized_tqa_gen,
                                    tokenized_tqa_gen_end_q)


def get_activation(cfg: ActivationConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map="auto")
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
    print("Tokenizing prompts")
    if cfg.dataset_name == "tqa_gen" or cfg.dataset_name == "tqa_gen_end_q":
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(
            f"../features/{cfg.model_name}_{cfg.dataset_name}_categories.pkl", "wb"
        ) as f:
            pickle.dump(categories, f)
    else:
        prompts, labels = formatter(dataset, tokenizer)
