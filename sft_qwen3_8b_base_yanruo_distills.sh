set -ex
WANDB_API_KEY=fae57cc134a5496320355cef38073633aaa454f8
uv run --prerelease=allow --active llamafactory-cli train examples/train_full/qwen3_8b_full_sft_yanruo_distills.yaml