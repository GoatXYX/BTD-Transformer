from torchtext.datasets import WikiText2
import os

# 指定保存目录
save_dir = "wikitext-2"
os.makedirs(save_dir, exist_ok=True)

# 下载并保存数据
for split in ["train", "valid", "test"]:
    with open(os.path.join(save_dir, f"{split}.txt"), "w") as f:
        for line in WikiText2(split=split):
            f.write(line + "\n")

print(f"WikiText-2 数据集已保存到目录 {save_dir} 中！")
