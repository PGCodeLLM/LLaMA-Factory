set -ex
export WANDB_API_KEY=38c8d9411c10a9205849607753c6585cc627a114
export DISABLE_VERSION_CHECK=1
uv run --prerelease=allow --active llamafactory-cli train examples/train_full/open_pangu_full_sft.yaml