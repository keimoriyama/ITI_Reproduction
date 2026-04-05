import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    pl.read_csv(
        "results_dump/answer_dump/meta-llama_Llama-2-7b-chat-hf_seed_42_top_32_heads_alpha_15_fold_0.csv"
    ).select(pl.col("Question", "meta-llama/Llama-2-7b-chat-hf"))
    return


@app.cell
def _(pl):
    pl.read_csv("results_dump/answer_dump/meta-llama_Llama-2-7b-chat-hf_seed_42_top_32_heads_alpha_15_fold_1.csv")
    return


@app.cell
def _(pl):
    pl.read_csv("results_dump/summary_dump/meta-llama_Llama-2-7b-chat-hf_seed_42_top_32_heads_alpha_15_fold_0.csv")
    return


if __name__ == "__main__":
    app.run()
