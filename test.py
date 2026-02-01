from datasets import load_dataset

ds = load_dataset("JoeYing/ReTool-SFT", split="train")

def check_invalid(example):
    msgs = example["messages"]
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if role not in {"user", "assistant", "system"}:
            return True
        if not isinstance(content, str) or not content.strip():
            return True
    return False

invalid_count = sum(1 for ex in ds if check_invalid(ex))
print("Invalid examples detected:", invalid_count)
