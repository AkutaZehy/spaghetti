import os
import torch

# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
# print("Current working directory:", cwd)

model_dir = os.path.abspath(os.path.join(cwd, '..', 'output'))

print(model_dir)

structure = 'defect_coco_re'

name = "model_0004999.pth"
# name = 'model_final.pth'
# name = 'ade20k_bottleneck1.pth'

model_path = os.path.abspath(os.path.join(model_dir,structure, name))

# print(model_path)

checkpoint = torch.load(model_path, map_location=torch.device("cpu"))  # 选择CPU避免显存占用

# 检查 checkpoint 的类型和结构
if isinstance(checkpoint, dict):
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
else:
    # 如果加载结果不是字典，则假定它是模型实例
    state_dict = checkpoint.state_dict()

# 确保 state_dict 是一个字典，并且有参数
if not isinstance(state_dict, dict) or len(state_dict) == 0:
    raise ValueError("加载的 state_dict 为空或格式不正确，请检查 pth 文件的保存方式。")

total_params = sum(param.numel() for param in state_dict.values())
print(f"Total parameters: {total_params}")