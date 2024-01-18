import torch
from src.mods.infer import Inference

def infer_main(ckpt_file, structrues_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    infer = Inference(ckpt_file, device)
    infer.inference(structrues_file)