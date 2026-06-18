import sys
import torch
import transformers
import datasets
import peft
import trl

print("====== Python 环境 ======")
print("python:", sys.version)

print("\n====== 核心库版本 ======")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("trl:", trl.__version__)

print("\n====== GPU 检查 ======")
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("GPU 名称:", torch.cuda.get_device_name(0))
else:
    print("当前没有可用 CUDA GPU。")
    print("你是 AMD 显卡，本地第一阶段先用 CPU / 小样本跑通流程。")

print("\n====== lm-eval 检查 ======")
try:
    import lm_eval
    print("lm_eval 可导入，安装成功。")
except Exception as e:
    print("lm_eval 导入失败:", repr(e))