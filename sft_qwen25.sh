WANDB_API_KEY=fae57cc134a5496320355cef38073633aaa454f8
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --prerelease=allow --active llamafactory-cli train examples/train_full/qwen2_5_instruct_7b_full_sft.yaml