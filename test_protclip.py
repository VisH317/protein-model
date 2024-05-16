import torch
from protCLIP.arch import ProtCLIP

if __name__ == "__main__":
    clip = ProtCLIP(16, 16, 8, 16)

    z = torch.rand(2, 2, 16)
    t = torch.rand(2, 2, 16)

    print(clip(z, t))
    