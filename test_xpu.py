import torch
print("torch:", torch.__version__)
import intel_extension_for_pytorch as ipex
print("IPEX:", ipex.__version__)
print("XPU:", torch.xpu.is_available())
if torch.xpu.is_available():
    print("Device:", torch.xpu.get_device_name(0))
