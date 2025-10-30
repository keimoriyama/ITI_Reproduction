import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import polars as pl
    return np, pl


@app.cell
def _(np):
    np.load('./features/meta-llama_Llama-2-7b-chat-hf_tqa_mc2_labels.npy').shape
    return


@app.cell
def _(np):
    np.load("./features/meta-llama_Llama-2-7b-chat-hf_tqa_mc2_layer_wise.npy")
    return


@app.cell
def _(pl):
    pl.read_csv("./features/meta-llama_Llama-2-7b-chat-hf_tqa_mc2_prompts.csv")
    return


if __name__ == "__main__":
    app.run()
