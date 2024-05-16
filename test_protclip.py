import torch
from protCLIP.arch import ProtCLIP
import time

if __name__ == "__main__":

    clip = ProtCLIP(16, 16, 8, 16)
    crit = torch.nn.BCELoss()
    opt = torch.optim.AdamW(clip.parameters())

    while True:

        opt.zero_grad()
        z = torch.rand(2, 2, 16)
        t = torch.rand(2, 2, 16)

        out = clip(z, t)

        # crit.register_backward_hook(lambda grad: print(grad))
        loss = crit(out, torch.as_tensor([[1, 0.5], [0.5, 0]]))

        print(out)

        loss.backward()
        opt.step()


        time.sleep(2)

    