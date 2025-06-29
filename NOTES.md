## Welcome to my hackathon notes 

Some important things from the GPU puzzles I've learned... 




# Kernel Comp...

1. Register with: `@compiler.register('cond1d')
2. Compile with: mojo package op -o op.mojopkg
3. Load ops: ops = CustomOpLibrary(Path("op"))
4. Pre-allocated tensor: output = torhc.empty_like(input_tensor)
5. Call conv1d = ops.conv1d


## Conv2d tiled 

Each tile handles TPB output pixels.

grid_dim = (output_width + TPB_X - 1) // TPB_X, out_chan, B)

The grid's first dimension is how many tiles we need to 
cover the output_width.
The grid's second dim gives each output channel a tile set 

Each thread-block:
    1. Copyies a patch of input into shared
    2. Copies kerenl filter into shared
    3. Loops over all of in_channels 
