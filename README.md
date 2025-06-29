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
- [x] Write a  working conv2d kernel with stride and padding
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


## Attributes to focus on
- [x] Project Compiles
- [x] Results theoritically expected..
- [ ] Results correct 


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


