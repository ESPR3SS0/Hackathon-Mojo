# Welcome to my Project 

The goals were loosely...
1. Write a `bmm` kernel that 'works'
2. Write a `conv2d` kernel that 'works'
3. Write a `conv2d` kernel with stride and paddign that 'works'
4. Inject an existing torch model with the `bmm` kernel
5. Inject an existing torch model with the `bmm` and `conv2d` kernel

Extended goals:
1. Write a `conv2d` kernel that is effecient for unstructure-ly pruned conv2d and linear layers 

The status at the end...
- [x] Write basic BMM kernel that worksmm 
- [x] Write a  working conv2d kernel (superceded by stride+padding version)
- [x] Write a  Kinda working conv2d kernel with stride and padding (atol on order of 1e-2 D:)
- [ ] Write a  FULLY working conv2d kernel with stride and padding
- [x] Script to 'inject' `bmm` into pretrained net 
- [ ] Script to 'inject' `conv2d` into pretrained net 
- [ ] Sparse kernels

# Quick Start

Setup Env
```basj
pixi shell
```

Build Kernels (bmm and conv2d_pad)
```bash
mojo package kernels -o kernels.mojopkg
```

BenchMark
```bash
pixi run kernel-comp
pixi run bench-bmm
pixi run bench-conv2d
```

```bash
pixi shell
mojo package kernels -o kernels.mojopkg
python bench.py bmm
python bench.py conv2d
```

Run a IMAGENET model and inject the custom kernels
```bash
python injector.py
```


## Results

Injector
```
Inferencing on all models
Checking for matching values
Torch and bmm mojo match!
Original  : 4981.794261932373 ms / iter
Mojo-BMM  : 24243.74566078186 ms / iter
```

Conv bench
```
=== Mojo vs PyTorch ‒ latency statistics (ms) ===

Op                         mean-Pt   med-Pt   min-Pt   max-Pt    mean-Mo   med-Mo   min-Mo   max-Mo   Med Speed-up
[PASSED] Test 0               0.08     0.07     0.07     0.20     115.65    96.63    93.07   181.15         0.00×
[PASSED] Test 1               0.14     0.13     0.12     0.30     126.19   102.36   100.25   220.34         0.00×
[PASSED] Test 2               0.15     0.14     0.14     0.29     124.95   105.99   100.99   186.51         0.00×
[FAILED] Test 3 | [DESC] B:32 | in_channels:16 | out_channels:32 | h:64 | w:32 | ker_h:6 | ker_w:6 | pad_w:0 | pad_h:0 | stride_w:2 | stride_h:2  0.22     0.21     0.21     0.35     123.34   103.63   100.62   184.08         0.00×
[FAILED] Test 4 | [DESC] B:32 | in_channels:16 | out_channels:32 | h:64 | w:32 | ker_h:6 | ker_w:6 | pad_w:3 | pad_h:3 | stride_w:2 | stride_h:2  0.15     0.15     0.14     0.28     130.87   110.30   104.86   208.31         0.00×
[PASSED] Test 5               0.11     0.10     0.10     0.21     126.17   103.94    99.82   206.69         0.00×
[PASSED] Test 6               0.16     0.15     0.15     0.33     131.79   111.39   104.87   208.56         0.00×
Passed: 5 | Failed: 2
```

BMM bench
```
=== Mojo vs PyTorch ‒ latency statistics (ms) ===

Op                         mean-Pt   med-Pt   min-Pt   max-Pt    mean-Mo   med-Mo   min-Mo   max-Mo   Med Speed-up
[PASSED] Test 0 | A:torch.Size([32, 512, 512]) B:torch.Size([32, 512, 512])     0.90     0.89     0.89     1.01     128.40   107.06   102.77
262.02         0.01×
[PASSED] Test 1 | A:torch.Size([2, 1, 512]) B:torch.Size([2, 512, 1000])     0.05     0.03     0.03     0.23     128.96   103.89    97.56   257.340.00×
[PASSED] Test 2 | A:torch.Size([2, 1, 512]) B:torch.Size([2, 512, 10])     0.06     0.05     0.04     0.20     125.94   111.72    92.36   192.64  0.00×
[PASSED] Test 3 | A:torch.Size([2, 512, 32]) B:torch.Size([2, 32, 16])     0.05     0.04     0.04     0.20     119.09   100.35    93.61   183.32  0.00×
```


## Mini Write Up

I was very close to having `conv2d` work with stride and padding. While it works 
when padding=0 and stride=1, and even in some cases where padding!=0 and 
stride!=0, it is not working in every case. The logic for loading the data 
gets tedious (but absolutely doable) once there are so many different 'divisions'
of your data (i.e. tiles, needing offsets, getting the write window given 
stride and offset). 

The BMM was much more striaght forward. I would've liked to use some of the fun 
functional patterns that were introduce in the later GPU puzzles, but I have 
yet to play with those enough to quickly leverage them. 

The model injector was very fun. It simply loads the torch opt, but something 
about the ability to generate a kernel in a langauge as nice as mojo, and 
inject it into an existing model is very cool to me. I hope to soon implement 
this type of idea in a Repo that's dedictated to speeding up inference via 
pruning, quantization, and now _custom kernels_. Without mojo's kernel syntax,
that would not have been in my toolbox! 

**Impact:** As a result of this project, I have two new kernels `bmm` and `conv2d`. 

**Remaining work:** I _badly_ want to improve bmm, and use some of the functional gpu programmign syntax. The `conv2d` kernel is not numerically stable (I _believe_ it's just numeric stability holding it back and not logic...). As it can be seen in by the benchmarks, the kernels themselves are relatively quite slow, so there's many optimizatoins on the table. 

**Extension:** Eventually, once a good `conv2d` kernel is fleshed out, I want to write a `sparse_conv2d` kernel for unstructured-ly pruned layers. 

**Extension** If I fleshed out soem more layers, maxpool, relu, etc, I could fully inject an existing troch model with kernels... or run a Max graph. Either way that would be super cool.


# BMM Kernel! 

The kernel is very very vanilla. It takes two input matrices, batch mat muls 's them. 


# Conv2d Kernel 
To the best of my knowledge this kernel supports:
- padding 
- stride 

The tests will show "

# Injection to IMAGENET





## Init brainstrom


I have some ideas of projects that correlated well with my research. Generally:
1. Kernels for model that have been unstructurally pruned 
2. Hand written fused kernels for resnet like model
3. Kernels fo structured pruned models 


The recommended projects are: 
1. Batched Matrix Multiplication (BMM)
2. Multi-head latent attention 
3. Mixture of experts 
4. Non-maximum supression
5. Group matrix multiplication
6. 2D convolutions
7. Hopper general matrix vector muliply

Or implement these models with the MAX graph API
1. Whisper 
2. YOLO
3. SAM or MobileSAM
4. Bagel-7B
5. Generative Recommenders


