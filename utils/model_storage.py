import os
import torch

def save_checkpoint(model, epoch, model_path,iteration,prefix =""):
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    model_out_path = model_path + prefix + "model_epoch_{}".format(epoch) + ".pth"
    state = {"epoch":epoch, "model":model}
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state,model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))