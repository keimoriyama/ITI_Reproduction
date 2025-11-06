import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import polars as pl
    from transformers import AutoModelForCausalLM
    from einops import rearrange
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import seaborn as sns
    import random


    import gc

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", device_map="auto", trust_remote_code=True
    )
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    del model
    gc.collect()
    num_layers, num_heads
    return (
        LogisticRegression,
        accuracy_score,
        np,
        num_heads,
        num_layers,
        pl,
        random,
        rearrange,
        sns,
    )


@app.cell
def _(np, num_heads, rearrange):
    labels = np.load('./features/meta-llama_Llama-2-7b-chat-hf_tqa_mc2_labels.npy')
    layer_wise_activation = np.load("./features/meta-llama_Llama-2-7b-chat-hf_tqa_mc2_layer_wise.npy")
    head_wise_activation = np.load("./features/meta-llama_Llama-2-7b-chat-hf_tqa_mc2_head_wise.npy")
    head_wise_activation = rearrange(
        head_wise_activation, "b l (h d) -> b l h d", h=num_heads
    )
    labels.shape, head_wise_activation.shape, layer_wise_activation.shape
    return head_wise_activation, labels


@app.cell
def _(pl):
    pl.read_csv("./features/meta-llama_Llama-2-7b-chat-hf_tqa_mc2_prompts.csv")
    return


@app.cell
def _(head_wise_activation, labels, random):
    data_num = head_wise_activation.shape[0]
    train_idx = random.sample(range(data_num), k=int(data_num * 0.75))
    val_idx = list(set(range(data_num)) - set(train_idx))
    all_X_train = head_wise_activation[train_idx]
    y_train = labels[train_idx]
    all_X_val = head_wise_activation[val_idx]
    y_val = labels[val_idx]
    all_X_train.shape, y_train.shape
    return all_X_train, all_X_val, y_train, y_val


@app.cell
def _(all_X_train, y_train):
    all_X_train.shape, y_train.shape
    return


@app.cell
def _(
    LogisticRegression,
    accuracy_score,
    all_X_train,
    all_X_val,
    num_heads,
    num_layers,
    y_train,
    y_val,
):
    accs = []
    for layer in range(num_layers):
        for head in range(num_heads):
            X_train = all_X_train[:, layer, head, :]
            X_val = all_X_val[::, layer, head, :]
            clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            accs.append({"layer": layer, "head": head, "accuracy": acc})
    return (accs,)


@app.cell
def _(accs, pl):
    pl.DataFrame(accs).pivot(
        "head", index="layer", values="accuracy"
    ).to_pandas().set_index(
        "layer"
    ).sort_index(ascending=False)
    return


@app.cell
def _(accs, pl, sns):
    sns.heatmap(
        pl.DataFrame(accs)
        .pivot("head", index="layer", values="accuracy")
        .to_pandas()
        .set_index("layer")
        .sort_index(ascending=False),
        annot=False,
        cmap="Blues",
    )
    return


if __name__ == "__main__":
    app.run()
