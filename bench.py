import time
import random
import numpy as np
from pathlib import Path
from max.driver import CPU, Accelerator, Device, Tensor, accelerator_count
from art import text2art
from max.engine import InferenceSession
import torch
from alive_progress import alive_it
import torch.nn.functional as F
from max.torch import CustomOpLibrary  # pip install modular
from cyclopts import App
from rich.console import Console

from statistics import mean, median, multimode   # add to imports
app = App()
console = Console()

PKG_PATH = Path(__file__).with_name("kernels.mojopkg")
ops = CustomOpLibrary(PKG_PATH)

DEBUG = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


LEARNED_THINGS = [
    "POST-compile errors look scarier than they are",
    "The conv1d example has mut=True on purpose... imagine that!",
    "[comp time](args)",
    "Dont forget Turn off grad for custom ops",
    "Always remember your offsets!",
    "tiles, patches, blocks, warps, blocks, grids, everythings a block",
]

@torch.compile
def mojo_bmm(A, B, out):

    #mojo_conv2d = ops.bmm_tiled[{}]
    ops.bmm_tiled(out, A, B)
    return out


def get_mojo_conv2d(
    B,
    in_channels,
    out_channels,
    h,
    w,
    ker_h,
    ker_w,
    pad_w,
    pad_h,
    stride_w,
    stride_h,
):
    ker_params = {
        "batch": B,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "H": h,
        "W": w,
        "ker_h": ker_h,
        "ker_w": ker_w,
        "pad_w": pad_w,
        "pad_h": pad_h,
        "stride_w": stride_w,
        "stride_h": stride_h,
    }

    spec_mojo_conv2d = ops.conv2d_pad[ker_params]
    mojo_conv2d= torch.compile(spec_mojo_conv2d)#(out_t, inp_t, ker_t)
    return mojo_conv2d





def mojo_conv2d_pad_torch_op(
    B,
    in_channels,
    out_channels,
    h,
    w,
    ker_h,
    ker_w,
    out_t,
    inp_t,
    ker_t,
    pad_w,
    pad_h,
    stride_w,
    stride_h,
):
    ker_params = {
        "batch": B,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "H": h,
        "W": w,
        "ker_h": ker_h,
        "ker_w": ker_w,
        "pad_w": pad_w,
        "pad_h": pad_h,
        "stride_w": stride_w,
        "stride_h": stride_h,
    }

    mojo_conv2d = ops.conv2d_pad[ker_params]
    torch.compile(mojo_conv2d)(out_t, inp_t, ker_t)
    torch.cuda.synchronize()
    return out_t


def mojo_conv2d_torch_op(
    B, in_channels, out_channels, h, w, ker_h, ker_w, out_t, inp_t, ker_t
):
    ker_params = {
        "batch": B,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "H": h,
        "W": w,
        "ker_h": ker_h,
        "ker_w": ker_w,
    }
    mojo_conv2d = ops.conv2d[ker_params]
    torch.compile(mojo_conv2d)(out_t, inp_t, ker_t)
    torch.cuda.synchronize()
    return out_t

def compile_and_run_mojo_bmm(A: torch.Tensor, B: torch.Tensor):
    """(B,M,K) @ (B,K,N) → (B,M,N) via Mojo kernel."""
    Bsz, M, K = A.shape
    N = B.shape[2]
    C = torch.empty((Bsz, M, N), device=A.device, dtype=A.dtype)

    bmm = ops.bmm_tiled[{}]
    torch.compile(bmm)(C,A,B)
    return C


def mojo_bmm_torch_op(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, compiled_fn: None) -> torch.Tensor:
    """(B,M,K) @ (B,K,N) → (B,M,N) via Mojo kernel."""
    return compiled_fn(C, A, B)


#jldef timeit_mojo_fn(mojo_fn, A, B, C, warm=5, rep=20) -> float:
#def timeit_mojo_fn(mojo_fn, *params, warm=5, rep=20) -> float:
#    """Return median milliseconds for `fn()` on current torch DEVICE."""
#
#    torch.cuda.synchronize() if torch.cuda.is_available() else None
#
#    for _ in range(warm):
#        mojo_fn(*params)
#    torch.cuda.synchronize() if torch.cuda.is_available() else None
#
#    timings = []
#
#    #for _ in alive_it(range(rep), title=random.choice(LEARNED_THINGS)):
#    for _ in range(rep):
#        t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
#        t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
#
#        if t0:
#            t0.record()
#            out = mojo_fn(*params)
#            t1.record()
#            torch.cuda.synchronize()
#            timings.append(t0.elapsed_time(t1))  # ms
#        else:
#            start = time.perf_counter()
#            out = mojo_fn(*params)
#            timings.append((time.perf_counter() - start) * 1e3)
#
#    return sorted(timings)[rep // 2], out  # median,
#
#
#
#
#
#def timeit(fn, warm=5, rep=20) -> float:
#    """Return median milliseconds for `fn()` on current torch DEVICE."""
#
#    torch.cuda.synchronize() if torch.cuda.is_available() else None
#
#    for _ in range(warm):
#        fn()
#    torch.cuda.synchronize() if torch.cuda.is_available() else None
#
#    timings = []
#
#    #for _ in alive_it(range(rep), title=random.choice(LEARNED_THINGS)):
#    for _ in range(rep):
#        t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
#        t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
#
#        if t0:
#            t0.record()
#            fn()
#            t1.record()
#            torch.cuda.synchronize()
#            timings.append(t0.elapsed_time(t1))  # ms
#        else:
#            start = time.perf_counter()
#            fn()
#            timings.append((time.perf_counter() - start) * 1e3)
#
#    return sorted(timings)[rep // 2]# median,


@app.command()
def bmm():
    """
    Bench BMM
    """

    def test1():
        """square"""
        B, M, K, N = 32, 512, 512, 512
        A = torch.randn(B, M, K, device=DEVICE, dtype=torch.float32)
        X = torch.randn(B, K, N, device=DEVICE)
        C = torch.empty((B, M, N), device=A.device, dtype=A.dtype)
        expected = torch.bmm(A, X)
        return (A, X, C, expected)

    def test2():
        """A short n wide"""
        B, M, K, N = 2, 1, 512, 1000
        A = torch.randn(B, M, K, device=DEVICE, dtype=torch.float32)
        X = torch.randn(B, K, N, device=DEVICE)
        C = torch.empty((B, M, N), device=A.device, dtype=A.dtype)
        expected = torch.bmm(A, X)
        return (A, X, C, expected)

    def test3():
        """A short n wide, B short n wide"""
        B, M, K, N = 2, 1, 512, 10
        A = torch.randn(B, M, K, device=DEVICE, dtype=torch.float32)
        X = torch.randn(B, K, N, device=DEVICE)
        C = torch.empty((B, M, N), device=A.device, dtype=A.dtype)
        expected = torch.bmm(A, X)
        return (A, X, C, expected)

    def test4():
        """A tall n skinny"""
        B, M, K, N = 2, 512, 32, 16
        A = torch.randn(B, M, K, device=DEVICE, dtype=torch.float32)
        X = torch.randn(B, K, N, device=DEVICE)
        C = torch.empty((B, M, N), device=A.device, dtype=A.dtype)
        expected = torch.bmm(A, X)
        return (A, X, C, expected)

    #tests = [test1(), test2(), test3(), test4(), test5()]
    tests = [test1(), test2(), test3(), test4()]
    results = []

    # Get the compiled troch op 
    #bmm = ops.bmm_tiled[{}]
    #mojo_bmm = torch.compile(bmm)

    for i, params in enumerate(tests):
        #t_mojo, m_out = timeit(lambda: mojo_bmm_torch_op(params[0], params[1]))
        A, B, C = params[0], params[1], params[2]

        #mojo_bmm(A, B, C)
        mojo_out = mojo_bmm(A,B,C)
        #mojo_out = compile_and_run_mojo_bmm(A, B)
        torch_out = torch.bmm(A,B)

        mojo_stats = timeit(lambda: mojo_bmm(C, A, B))
        torch_stats = timeit(lambda: torch.bmm(A, B))

        if torch.allclose(mojo_out, torch_out, atol=1e-4):
            results.append((f"[PASSED] Test {i} | A:{A.shape} B:{B.shape}", torch_stats, mojo_stats))
        else:
            results.append((f"[FAILED] Test {i} | A:{A.shape} B:{B.shape}", torch_stats, mojo_stats))

    report(results)


    return

def _fmt(m):   # helper – keep numbers tidy
    return f"{m:8.2f}"

def _fmt(m):   # helper – keep numbers tidy
    return f"{m:8.2f}"

def report(results):
    console.print("=== Mojo vs PyTorch ‒ latency statistics (ms) ===\n")
    hdr = (
        "Op",               "mean-Pt", "med-Pt",  "min-Pt", "max-Pt",
                          "mean-Mo", "med-Mo",  "min-Mo", "max-Mo",
        "Med Speed-up"
    )
    console.print("{:<25} {:>8} {:>8} {:>8} {:>8}   "
                  "{:>8} {:>8} {:>8} {:>8}   {:>10}"
                  .format(*hdr))

    for name, pt, mo in results:
        pt_mean, pt_med, pt_mode, pt_min, pt_max = pt
        mo_mean, mo_med, mo_mode, mo_min, mo_max = mo
        speed = pt_med / mo_med if mo_med else float("nan")
        console.print(
            f"{name:<25} "
            f"{_fmt(pt_mean)} {_fmt(pt_med)} {_fmt(pt_min)} {_fmt(pt_max)}   "
            f"{_fmt(mo_mean)} {_fmt(mo_med)} {_fmt(mo_min)} {_fmt(mo_max)}   "
            f"{speed:10.2f}×"
        )


#def report(results):
#    console.print("=== Mojo vs PyTorch – median latency (ms) ===")
#    console.print("{:<20} {:>12} {:>12} {:>9}".format("Op", "PyTorch", "Mojo", "Pt/Mojo"))
#    for name, t_pt, t_mo in results:
#        speed =  t_pt / t_mo if t_mo else float("nan")
#        console.print(f"{name:<20} {t_pt:>10.2f}   {t_mo:>10.2f}   {speed:>8.2f}×")
#

@app.command()
def conv2d():


    def test7():
        """weirdest"""
        hparams = {
            "B": 32,
            "in_channels": 16,
            "out_channels": 32,
            "h": 64,
            "w": 32,
            "ker_h": 6,
            "ker_w": 3,
            "pad_w": 2,
            "pad_h": 0,
            "stride_w": 2,
            "stride_h": 1,
        }
        dtype = torch.float32

        # Make sure all libs see the same seed
        torch.manual_seed(0)
        np.random.seed(0)

        inp_t = torch.randn(
            hparams["B"],
            hparams["in_channels"],
            hparams["h"],
            hparams["w"],
            device=DEVICE,
            dtype=dtype,
        )
        ker_t = torch.randn(
            hparams["out_channels"],
            hparams["in_channels"],
            hparams["ker_h"],
            hparams["ker_w"],
            device=DEVICE,
            dtype=dtype,
        )
        # out_t  = torch.empty(hparams['B'], hparams['out_channels'], hparams['h']-hparams['ker_h']+1, hparams['w']-hparams['ker_w']+1, device=DEVICE,   dtype=dtype)

        # expected = F.conv2d(inp_t, ker_t, None, stride=(hparams['stride_h'],hparams['stride_w'],  )
        torch_conv = torch.nn.Conv2d(
            hparams["in_channels"],
            hparams["out_channels"],
            (hparams["ker_h"], hparams["ker_w"]),
            stride=(hparams["stride_h"], hparams["stride_w"]),
            bias=False,
            padding=(hparams["pad_h"], hparams["pad_w"]),
        ).to(DEVICE, dtype)
        torch_conv.weight.data.copy_(ker_t)
        # expected =  torch_conv(inp_t)
        return (inp_t, ker_t, hparams, torch_conv)



    def test6():
        """weirdest"""
        hparams = {
            "B": 32,
            "in_channels": 16,
            "out_channels": 32,
            "h": 64,
            "w": 32,
            "ker_h": 6,
            "ker_w": 3,
            "pad_w": 2,
            "pad_h": 0,
            "stride_w": 2,
            "stride_h": 2,
        }
        dtype = torch.float32

        # Make sure all libs see the same seed
        torch.manual_seed(0)
        np.random.seed(0)

        inp_t = torch.randn(
            hparams["B"],
            hparams["in_channels"],
            hparams["h"],
            hparams["w"],
            device=DEVICE,
            dtype=dtype,
        )
        ker_t = torch.randn(
            hparams["out_channels"],
            hparams["in_channels"],
            hparams["ker_h"],
            hparams["ker_w"],
            device=DEVICE,
            dtype=dtype,
        )
        # out_t  = torch.empty(hparams['B'], hparams['out_channels'], hparams['h']-hparams['ker_h']+1, hparams['w']-hparams['ker_w']+1, device=DEVICE,   dtype=dtype)

        # expected = F.conv2d(inp_t, ker_t, None, stride=(hparams['stride_h'],hparams['stride_w'],  )
        torch_conv = torch.nn.Conv2d(
            hparams["in_channels"],
            hparams["out_channels"],
            (hparams["ker_h"], hparams["ker_w"]),
            stride=(hparams["stride_h"], hparams["stride_w"]),
            bias=False,
            padding=(hparams["pad_h"], hparams["pad_w"]),
        ).to(DEVICE, dtype)
        torch_conv.weight.data.copy_(ker_t)
        # expected =  torch_conv(inp_t)
        return (inp_t, ker_t, hparams, torch_conv)



    def test5():
        """non square kernel, stride"""
        hparams = {
            "B": 32,
            "in_channels": 16,
            "out_channels": 32,
            "h": 64,
            "w": 32,
            "ker_h": 6,
            "ker_w": 6,
            "pad_w": 3,
            "pad_h": 3,
            "stride_w": 2,
            "stride_h": 2,
        }
        dtype = torch.float32

        # Make sure all libs see the same seed
        torch.manual_seed(0)
        np.random.seed(0)

        inp_t = torch.randn(
            hparams["B"],
            hparams["in_channels"],
            hparams["h"],
            hparams["w"],
            device=DEVICE,
            dtype=dtype,
        )
        ker_t = torch.randn(
            hparams["out_channels"],
            hparams["in_channels"],
            hparams["ker_h"],
            hparams["ker_w"],
            device=DEVICE,
            dtype=dtype,
        )

        torch_conv = torch.nn.Conv2d(
            hparams["in_channels"],
            hparams["out_channels"],
            (hparams["ker_h"], hparams["ker_w"]),
            stride=(hparams["stride_h"], hparams["stride_w"]),
            bias=False,
            padding=(hparams["pad_h"], hparams["pad_w"]),
        ).to(DEVICE, dtype)
        torch_conv.weight.data.copy_(ker_t)
        return (inp_t, ker_t, hparams, torch_conv)




    def test4():
        """non square kernel, stride"""
        hparams = {
            "B": 32,
            "in_channels": 16,
            "out_channels": 32,
            "h": 64,
            "w": 32,
            "ker_h": 6,
            "ker_w": 6,
            "pad_w": 0,
            "pad_h": 0,
            "stride_w": 2,
            "stride_h": 2,
        }
        dtype = torch.float32

        # Make sure all libs see the same seed
        torch.manual_seed(0)
        np.random.seed(0)

        inp_t = torch.randn(
            hparams["B"],
            hparams["in_channels"],
            hparams["h"],
            hparams["w"],
            device=DEVICE,
            dtype=dtype,
        )
        ker_t = torch.randn(
            hparams["out_channels"],
            hparams["in_channels"],
            hparams["ker_h"],
            hparams["ker_w"],
            device=DEVICE,
            dtype=dtype,
        )
        # out_t  = torch.empty(hparams['B'], hparams['out_channels'], hparams['h']-hparams['ker_h']+1, hparams['w']-hparams['ker_w']+1, device=DEVICE,   dtype=dtype)

        # expected = F.conv2d(inp_t, ker_t, None, stride=(hparams['stride_h'],hparams['stride_w'],  )
        torch_conv = torch.nn.Conv2d(
            hparams["in_channels"],
            hparams["out_channels"],
            (hparams["ker_h"], hparams["ker_w"]),
            stride=(hparams["stride_h"], hparams["stride_w"]),
            bias=False,
            padding=(hparams["pad_h"], hparams["pad_w"]),
        ).to(DEVICE, dtype)
        torch_conv.weight.data.copy_(ker_t)
        # expected =  torch_conv(inp_t)
        return (inp_t, ker_t, hparams, torch_conv)

    def test3():
        """non square inp, padding"""
        hparams = {
            "B": 32,
            "in_channels": 16,
            "out_channels": 32,
            "h": 64,
            "w": 32,
            "ker_h": 3,
            "ker_w": 3,
            "pad_w": 2,
            "pad_h": 2,
            "stride_w": 1,
            "stride_h": 1,
        }
        dtype = torch.float32

        # Make sure all libs see the same seed
        torch.manual_seed(0)
        np.random.seed(0)

        inp_t = torch.randn(
            hparams["B"],
            hparams["in_channels"],
            hparams["h"],
            hparams["w"],
            device=DEVICE,
            dtype=dtype,
        )
        ker_t = torch.randn(
            hparams["out_channels"],
            hparams["in_channels"],
            hparams["ker_h"],
            hparams["ker_w"],
            device=DEVICE,
            dtype=dtype,
        )

        torch_conv = torch.nn.Conv2d(
            hparams["in_channels"],
            hparams["out_channels"],
            (hparams["ker_h"], hparams["ker_w"]),
            stride=(hparams["stride_h"], hparams["stride_w"]),
            bias=False,
            padding=(hparams["pad_h"], hparams["pad_w"]),
        ).to(DEVICE, dtype)
        torch_conv.weight.data.copy_(ker_t)
        return (inp_t, ker_t, hparams, torch_conv)

    def test2():
        """non square inp"""
        hparams = {
            "B": 32,
            "in_channels": 16,
            "out_channels": 32,
            "h": 64,
            "w": 32,
            "ker_h": 3,
            "ker_w": 3,
            "pad_w": 0,
            "pad_h": 0,
            "stride_w": 1,
            "stride_h": 1,
        }
        dtype = torch.float32

        # Make sure all libs see the same seed
        torch.manual_seed(0)
        np.random.seed(0)

        inp_t = torch.randn(
            hparams["B"],
            hparams["in_channels"],
            hparams["h"],
            hparams["w"],
            device=DEVICE,
            dtype=dtype,
        )
        ker_t = torch.randn(
            hparams["out_channels"],
            hparams["in_channels"],
            hparams["ker_h"],
            hparams["ker_w"],
            device=DEVICE,
            dtype=dtype,
        )

        torch_conv = torch.nn.Conv2d(
            hparams["in_channels"],
            hparams["out_channels"],
            (hparams["ker_h"], hparams["ker_w"]),
            stride=(hparams["stride_h"], hparams["stride_w"]),
            bias=False,
            padding=(hparams["pad_h"], hparams["pad_w"]),
        ).to(DEVICE, dtype)
        torch_conv.weight.data.copy_(ker_t)
        return (inp_t, ker_t, hparams, torch_conv)

    def test1():
        hparams = {
            "B": 16,
            "in_channels": 3,
            "out_channels": 32,
            "h": 64,
            "w": 64,
            "ker_h": 3,
            "ker_w": 3,
            "pad_w": 0,
            "pad_h": 0,
            "stride_w": 1,
            "stride_h": 1,
        }
        dtype = torch.float32

        # Make sure all libs see the same seed
        torch.manual_seed(0)
        np.random.seed(0)

        inp_t = torch.randn(
            hparams["B"],
            hparams["in_channels"],
            hparams["h"],
            hparams["w"],
            device=DEVICE,
            dtype=dtype,
        )
        ker_t = torch.randn(
            hparams["out_channels"],
            hparams["in_channels"],
            hparams["ker_h"],
            hparams["ker_w"],
            device=DEVICE,
            dtype=dtype,
        )

        torch_conv = torch.nn.Conv2d(
            hparams["in_channels"],
            hparams["out_channels"],
            (hparams["ker_h"], hparams["ker_w"]),
            stride=(hparams["stride_h"], hparams["stride_w"]),
            bias=False,
            padding=(hparams["pad_h"], hparams["pad_w"]),
        ).to(DEVICE, dtype)
        torch_conv.weight.data.copy_(ker_t)
        return (inp_t, ker_t, hparams, torch_conv)

    atols = []
    results = []
    tests = [test1(), test2(), test3(), test4(), test5(), test6(), test7()]

    passed = 0
    failed = 0
    for i, params in enumerate(tests):
        inp_t = params[0]
        ker_t = params[1]
        hparams = params[2]
        torch_conv = params[3]

        H_out = (hparams["h"] + 2 * hparams["pad_h"] - hparams["ker_h"]) // hparams[
            "stride_h"
        ] + 1
        W_out = (hparams["w"] + 2 * hparams["pad_w"] - hparams["ker_w"]) // hparams[
            "stride_w"
        ] + 1

        out_t = torch.empty(
            (hparams["B"], hparams["out_channels"], H_out, W_out),
            device="cuda",
            dtype=torch.float32,
        )

        # Compile the kernel 
        mojo_conv2d = get_mojo_conv2d(**hparams)

        torch_out = torch_conv(inp_t)
        mojo_conv2d(out_t, inp_t, ker_t)
        mojo_out = out_t

        t_mojo = timeit(lambda: mojo_conv2d(out_t, inp_t, ker_t))
        t_torch= timeit(lambda: torch_conv(inp_t))


        descrip = " | ".join(f"{k}:{v}" for k,v in hparams.items())

        if torch.allclose(mojo_out, torch_out, atol=1e-4):
            if DEBUG:
                results.append((f"[PASSED] Test {i} | [DESC] {descrip}", t_torch, t_mojo))
            else:
                results.append((f"[PASSED] Test {i}", t_torch, t_mojo))

            passed+=1
        else:
            failed+=1
            results.append((f"[FAILED] Test {i} | [DESC] {descrip}", t_torch, t_mojo))
            if DEBUG:
                print("Max diff:", (mojo_out-torch_out).max())

            atols.append((torch_out - mojo_out).abs().max().item())

    report(results)
    console.print(f"Passed: {passed} | Failed: {failed}")
    console.print(f"The emp atols were: {atols}")
    return


def _summarise(timings):
    """Return (mean, median, mode, min, max) in milliseconds."""
    return (
        mean(timings),
        median(timings),
        multimode(timings)[0] if multimode(timings) else float("nan"),
        min(timings),
        max(timings),
    )

def timeit(fn, warm: int = 5, rep: int = 20):
    """Return (mean, median, mode, min, max) ms for `fn()` on current DEVICE."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(warm):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for _ in alive_it(range(rep), title=random.choice(LEARNED_THINGS)):
        if torch.cuda.is_available():
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            fn()
            t1.record()
            torch.cuda.synchronize()
            timings.append(t0.elapsed_time(t1))          # ms
        else:
            start = time.perf_counter()
            fn()
            timings.append((time.perf_counter() - start) * 1e3)

    return _summarise(timings)





if __name__ == "__main__":
    # banner = text2art("KERNEL BENCH", "random")
    # banner = text2art("KERNEL BENCH", "rnd-medium")
    banner = text2art("MOJO KERN BENCH", "rnd-small")
    console.print(banner, highlight=False)
    app()
