"""
File to inject mojo layers into pretrained models
"""

import time, torch, torchvision
from pathlib import Path
from max.torch import CustomOpLibrary

DEBUG = False

# Dat hand
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
])

valset = torchvision.datasets.CIFAR10(".", train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

PKG_PATH = Path(__file__).with_name("kernels.mojopkg")
ops = CustomOpLibrary(PKG_PATH)



PKG_PATH = Path(__file__).with_name("kernels.mojopkg")
ops = CustomOpLibrary(PKG_PATH)


def mojo_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """(B,M,K) @ (B,K,N) → (B,M,N) via Mojo kernel."""
    Bsz, M, K = A.shape

    N = B.shape[2]
    C = torch.empty((Bsz, M, N), device=A.device, dtype=A.dtype)

    # Compiler says this is needed 
    A = A.contiguous()
    B = B.contiguous()

    bmm = ops.bmm_tiled[{}]
    torch.compile(bmm)(C, A, B)

    return C


class BMMLinear(torch.nn.Module):
    """
    Inject the mojo kernel for forward passes
    """

    def __init__(self, linear):
        super().__init__()
        self.weight = torch.nn.Parameter(linear.weight.detach())
        self.bias   = torch.nn.Parameter(linear.bias.detach()) if linear.bias is not None else None

    def forward(self, x):
        B, K = x.shape
        N = self.weight.shape[0]
        A = x.unsqueeze(1)                    # (B,1,K
        W = self.weight.t().contiguous().unsqueeze(0)

        if B > 1:
            W = W.expand(B, -1, -1).contiguous()  # (B,K,N)

        out = mojo_bmm(A, W) #, 1, N, K)            # M=1
        out = out.squeeze(1)
        return out + self.bias if self.bias is not None else out

# ───────────────────────── Mojo Conv2d wrapper ──────────────────────────
def mojo_conv2d(x, w,  stride, padding, dilation=1, groups=1, b=None):
    """
    x: (N,C_in,H,W)  w: (C_out,C_in/groups,kH,kW)

    Currently supports stride=1, padding=none, dilation = int | tuple.

    Falls back to torch.nn.functional.conv2d when config is unsupported.
    """
    # Mojo kernel v1 supports only groups=1, symmetric stride/pad/dilation.
    #if groups != 1 or isinstance(stride, tuple) or isinstance(padding, tuple)\
    #   or isinstance(dilation, tuple):
    #    return F.conv2d(x, w, b, stride, padding, dilation, groups)

    #if stride is None:
    #    stride = (1,1)

    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape

    if DEBUG:
        print(f"Stride {stride}")
        print(f"padding {padding}")

    # Padding = (H, W)
    # Stride = (H, W)
    outH = (H + 2*padding[1] - dilation*(kH-1) - 1)//stride[1] + 1
    outW = (W + 2*padding[0] - dilation*(kW-1) - 1)//stride[0] + 1
    out  = torch.empty((N, C_out, outH, outW),
                       device=x.device, dtype=x.dtype)
    ker_params = {
        'batch'   : N,
        'in_channels' : C_in,
        'out_channels': C_out,
        'H'   : H,               
        'W'   : W,              
        'ker_h'  : kH,         
        'ker_w'  : kW,        
        'pad_w' : padding[0],
        'pad_h' : padding[1],
        'stride_w' : stride[0],
        'stride_h' : stride[1],
    }

    mojo_conv2d = ops.conv2d_pad[ker_params]
    torch.compile(mojo_conv2d)(out, x.contiguous(), w.contiguous())
    torch.cuda.synchronize()

    if b is not None:
        out += b.view(1, -1, 1, 1)
    return out

class MojoConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()

        self.weight = torch.nn.Parameter(conv.weight.detach(), requires_grad=False)
        self.bias   = torch.nn.Parameter(conv.bias.detach(), requires_grad=False) if conv.bias is not None else None
        # Save conv-specific hyper-params
        self.stride   = conv.stride   if isinstance(conv.stride, tuple) else conv.stride
        self.padding  = conv.padding  if isinstance(conv.padding, tuple) else conv.padding
        self.dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation
        self.groups   = conv.groups

    def forward(self, x):
        return mojo_conv2d(x, w=self.weight, b=self.bias,
                           stride=self.stride, padding=self.padding,
                           dilation=self.dilation, groups=self.groups)


def inject_mojo_ops_bmm(m):
    """
    Replace lin with bmm
    """
    for name, child in list(m.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(m, name, BMMLinear(child))
            #print("injected linear")
        else:
            inject_mojo_ops_bmm(child)



def inject_mojo_ops(m):
    """
    Replace lin with bmm
    """
    for name, child in list(m.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(m, name, BMMLinear(child))
            #print("injected linear")
        elif isinstance(child, torch.nn.Conv2d):
            setattr(m,name, MojoConv2d(child))
            #print("injected conv2d")
        else:
            inject_mojo_ops(child)


#def inject_mojo_ops_map(m, map):
#    """
#    Replace lin with bmm
#    """
#    for name, child in list(m.named_children()):
#        if child in map.keys():
#            setattr(m, name, map[m])
#            print(f"Injection {m} to {map[m]}")
#        else:
#            inject_mojo_ops_map(child, map)
#
#        #if isinstance(child, torch.nn.Linear):
#        #    setattr(m, name, BMMLinear(child))
        #    print("injected linear")
        #elif isinstance(child, torch.nn.Conv2d):
        #    setattr(m,name, MojoConv2d(child))
        #    print("injected conv2d")
        #else:
        #    inject_mojo_ops(child)


def bench(model, iters=5):
    """quck bench """
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            for img, _ in val_loader:
                model(img.cuda())
    torch.cuda.synchronize()

    return (time.time()-t0)*1000/iters   # ms / iteration





if __name__ == "__main__":

    # Define the injector map 
    map = {
        torch.nn.Linear : BMMLinear,
        torch.nn.Conv2d : MojoConv2d,
    }


    # Load the two model versions
    vanilla_model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().cuda()

    inj_model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().cuda()
    inject_mojo_ops(inj_model)


    bmm_model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval().cuda()
    inject_mojo_ops_bmm(bmm_model)

    print("Inferencing on all models")
    with torch.no_grad():
        x = torch.randn(4,3,224,224, device="cuda")
        out1 = vanilla_model(x)
        #out2 = inj_model(x)
        out3 = bmm_model(x)
        print(f"Checking for matching values")

        #if torch.allclose(out1, out2, atol=1e-4):
        #    print(f"Torch and conv+bmm mojo match!")
        #else:
        #    print("Mismatch between torch and conv_bmm mojo")

        if torch.allclose(out1, out3, atol=1e-5):
            print(f"Torch and bmm mojo match!")
        else:
            print("Mismatch between torch and bmm mojo")

    
    print("Original  :", bench(vanilla_model),   "ms / iter")
    #print("Mojo-BMM  :", bench(inj_model), "ms / iter")
    print("Mojo-BMM  :", bench(bmm_model), "ms / iter")


