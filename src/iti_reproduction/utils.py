import numpy as np
import torch


def format_truthfulqa(question, choice):
    return f"Q: {question} A: {choice}"


def format_truthfulqa_end_q(question, choice, rand_question):
    return f"Q: {question} A: {choice} Q: {rand_question}"


def tokenized_tqa(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    for i in range(len(dataset)):
        question = dataset[i]["question"]
        choices = dataset[i]["mc2_targets"]["choices"]
        labels = dataset[i]["mc2_targets"]["labels"]

        assert len(choices) == len(labels), (len(choices), len(labels))

        for j in range(len(choices)):
            choice = choices[j]
            label = labels[j]
            prompt = format_truthfulqa(question, choice)
            if i == 0 and j == 0:
                print(prompt)
            prompt = tokenizer(prompt, return_tensors="pt").input_ids
            all_prompts.append(prompt)
            all_labels.append(label)

    return all_prompts, all_labels


def tokenized_tqa_gen_end_q(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = dataset[i]["question"]
        category = dataset[i]["category"]
        rand_idx = np.random.randint(len(dataset))
        rand_question = dataset[rand_idx]["question"]

        for j in range(len(dataset[i]["correct_answers"])):
            answer = dataset[i]["correct_answers"][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors="pt").input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)

        for j in range(len(dataset[i]["incorrect_answers"])):
            answer = dataset[i]["incorrect_answers"][j]
            prompt = format_truthfulqa_end_q(question, answer, rand_question)
            prompt = tokenizer(prompt, return_tensors="pt").input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)

    return all_prompts, all_labels, all_categories


def tokenized_tqa_gen(dataset, tokenizer):
    all_prompts = []
    all_labels = []
    all_categories = []
    for i in range(len(dataset)):
        question = dataset[i]["question"]
        category = dataset[i]["category"]

        for j in range(len(dataset[i]["correct_answers"])):
            answer = dataset[i]["correct_answers"][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors="pt").input_ids
            all_prompts.append(prompt)
            all_labels.append(1)
            all_categories.append(category)

        for j in range(len(dataset[i]["incorrect_answers"])):
            answer = dataset[i]["incorrect_answers"][j]
            prompt = format_truthfulqa(question, answer)
            prompt = tokenizer(prompt, return_tensors="pt").input_ids
            all_prompts.append(prompt)
            all_labels.append(0)
            all_categories.append(category)

    return all_prompts, all_labels, all_categories


def get_llama_activations_pyvene(collected_model, collectors, prompt, device):
    with torch.no_grad():
        prompt = prompt.to(device)
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
    hidden_states = hidden_states.detach().cpu().numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    head_wise_hidden_states = (
        torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0)
        .squeeze()
        .numpy()
    )
    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states
