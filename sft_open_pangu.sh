set -ex
WANDB_API_KEY=fae57cc134a5496320355cef38073633aaa454f8
uv run --prerelease=allow --active llamafactory-cli train examples/train_full/open_pangu_full_sft.yaml