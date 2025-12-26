import cv2
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.cuda import cudart
import nvtx

from visionrt import compile
import visionrt.config as config


@nvtx.annotate("capture_overhead", color="green")
def capture_overhead(cap):
    ret, frame = cap.read()
    if not ret:
        return False
    return frame


@nvtx.annotate("preprocessing", color="purple")
def preprocessing(frame, memory_format=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).cuda().float()
    if memory_format:
        tensor = tensor.to(memory_format=memory_format)
    return tensor


def inference(tensor, model, name):
    with nvtx.annotate(f"inference_{name}", color="red"):
        _ = model(tensor)
        torch.cuda.synchronize()


def run_standard(cap, model, name, memory_format=None):
    with nvtx.annotate(f"standard_{name}", color="blue"):
        frame = capture_overhead(cap)
        if frame is False:
            return False
        tensor = preprocessing(frame, memory_format)
        inference(tensor, model, name)
    return True


def warmup(model, cap, iters=50, memory_format=None):
    """Warmup model before profiling."""
    for _ in range(iters):
        if not run_standard(cap, model, "warmup", memory_format):
            break


def profile(name, model, cap, iters, memory_format=None):
    """Profile model (call only after warmup and within profiler context)."""
    with nvtx.annotate(f"profile_{name}"):
        for _ in range(iters):
            if not run_standard(cap, model, name, memory_format):
                break


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FPS, 90)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    config.verbose = True

    profile_iters = int(os.environ.get("ITERS", "1000"))
    model = (
        nn.Sequential(
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        )
        .cuda()
        .eval()
    )

    cudart().cudaProfilerStart()  # type: ignore

    # base model
    warmup(model, cap)
    profile("baseline", model, cap, profile_iters)

    # base compiled mode
    torch._dynamo.reset()
    config.use_custom = False
    inductor_model = compile(copy.deepcopy(model))
    warmup(inductor_model, cap)
    profile("inductor", inductor_model, cap, profile_iters)
    del inductor_model

    # constant folding + kernel fusing model
    torch._dynamo.reset()
    config.optims.fold_conv_bn = True
    folding_fusing_model = compile(copy.deepcopy(model))
    warmup(folding_fusing_model, cap)
    profile("folding_fusing", folding_fusing_model, cap, profile_iters)
    config.optims.clear()
    del folding_fusing_model 

    # constant folding + kernel fusing + cudagraphs
    torch._dynamo.reset()
    config.optims.fold_conv_bn = True
    config.cudagraphs = True
    folding_fusing_cudagraph_model = compile(copy.deepcopy(model))
    warmup(folding_fusing_cudagraph_model, cap)
    profile("folding_fusing_cudagraph", folding_fusing_cudagraph_model, cap, profile_iters)
    config.optims.clear()
    config.cudagraphs = False
    del folding_fusing_cudagraph_model

    cudart().cudaProfilerStop()  # type: ignore
    cap.release()
