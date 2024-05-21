import torch
from torch import nn
from protCLIP.arch import ProtCLIP
import time

if __name__ == "__main__":

    clip = ProtCLIP(16, 16, 8, 16)
    c = torch.nn.CrossEntropyLoss()
    c2 = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(clip.parameters(), lr=0.01)

    z = torch.rand(2, 2, 16)
    t = torch.rand(2, 2, 16)
    z_ends = [1, 1]
    t_ends = [1, 1]

    while True:

        opt.zero_grad()

        out, out2 = clip(z, t, z_ends, t_ends)

        # crit.register_backward_hook(lambda grad: print(grad))
        target = torch.arange(2, dtype=torch.long)
        loss = (c(out, target) + c2(out2, target))/2

        print(out.softmax(dim=-1))
        print(loss)

        loss.backward()
        opt.step()


        time.sleep(2)

    