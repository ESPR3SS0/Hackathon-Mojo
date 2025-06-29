
#TODO: Copying and pasting from puzzles... could porlbably clean
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal
from gpu import thread_idx, block_idx, block_dim, barrier

# ANCHOR: conv1d_custom_op
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer



#alias TPB      = 32
# smaller to help numerical stability 
alias TPB      = 16
alias TPB_X    = TPB
alias TPB_Y    = TPB
alias THREADS_PER_BLOCK = (TPB_X, TPB_Y, 1)

alias dtype = DType.float32

fn conv2d_kernel[
    in_layout  : Layout,  # (B, C_in,  H,  W)
    out_layout : Layout,  # (B, C_out, H', W')
    k_layout   : Layout,  # (C_out, C_in, ker_h, ker_w)

    C_in       : Int,
    C_out      : Int,
    H          : Int,
    W          : Int,
    ker_h         : Int,
    ker_w         : Int,

    stride_h   : Int,
    stride_w   : Int,
    pad_h      : Int,
    pad_w      : Int,

    dtype      : DType = DType.float32,
](
    output : LayoutTensor[mut=True, dtype, out_layout],
    inp    : LayoutTensor[mut=True, dtype, in_layout],
    ker    : LayoutTensor[mut=True, dtype, k_layout],
):

    alias H_out = (H + 2*pad_h - ker_h) // stride_h + 1
    alias W_out = (W + 2*pad_w - ker_w) // stride_w + 1


    # Ouput tile base 
    # 
    # | b1| b2| ..| tile_x |
    # size * num blocks
    var gtile_x = block_idx.x * TPB_X
    var gtile_y = block_idx.y * TPB_Y

    #var bz  = block_idx.z

    # batch id, batch and out fold
    # batch id, it's folded with cout
    var n   = block_idx.z // C_out

    # out channel idx
    var out_channel_idx  = block_idx.z %  C_out

    var local_x = thread_idx.x
    var local_y = thread_idx.y

    # Global out x n y 
    var global_x = gtile_x + local_x
    var global_y = gtile_y + local_y

    # pixel height, pixel width
    alias SH = (TPB_Y - 1) * stride_h + ker_h
    alias SW = (TPB_X - 1) * stride_w + ker_w

    # Shared on inp patch. The hieght on wideth 
    var inp_tile = tb[dtype]().row_major[SH, SW]().shared().alloc()

    # Kernel is shared 
    var ker_tile = tb[dtype]().row_major[ker_h, ker_w]().shared().alloc()

    # I dont love this, but to be more stable I need 64
    #var acc : output.element_type = 0
    var acc : Float64 = 0

    # The stride need by ecah thread.
    var sy = local_y * stride_h
    var sx = local_x * stride_w

    for cur_in_channel in range(C_in):


        # FIRST - copy kenel weight for the curout, curint 
        if local_y < ker_h and local_x < ker_w:
            ker_tile[local_y, local_x] = ker[out_channel_idx, cur_in_channel, local_y, local_x]

        # 2) load the input patch this block needs
        #    base coord in input space for (global_y, global_x)
        var x_base = gtile_x * stride_w - pad_w + local_x * stride_w
        var y_base = gtile_y * stride_h - pad_h + local_y * stride_h

        # Need to load more elements because of the stride
        # main element
        for ry in range(stride_h):
            for rx in range(stride_w):

                # inp pach x and y 
                cur_x = x_base + rx
                cur_y = y_base + ry

                # tile x and y 
                tx = sx + rx
                ty = sy + ry

                # Gaurd for inp patch and load to tile
                if cur_x < W and cur_y < H:
                    inp_tile[ty, tx] = inp[n, cur_in_channel, cur_y, cur_x]
                else:
                    inp_tile[ty, tx] = 0



        ######## RIGHT 
        if local_x < ker_w - 1:
            var offset_x = stride_w * (TPB_X - ker_w) + ker_w

            var gx_r = offset_x + x_base
            var tile_x = sx + offset_x

            for ry in range(stride_h):
                for rx in range(stride_w):

                    cur_x = gx_r + rx
                    cur_y = y_base + ry

                    # Gaurd for inp patch
                    if cur_x < W and cur_y < H:
                        inp_tile[sy+ry, rx+tile_x] = inp[n, cur_in_channel, cur_y, cur_x]
                    else:
                        inp_tile[sy+ry, rx+tile_x] = 0

        ####### BOTTOM 
        if local_y < ker_h - 1:
            var offset_y = stride_h * (TPB_Y-ker_h) + ker_h
            var gy_b = y_base + offset_y
            var tile_y = sy + offset_y

            for ry in range(stride_h):
                for rx in range(stride_w):

                    cur_x = x_base + rx
                    cur_y = gy_b + ry

                    if cur_x < W and cur_y < H:
                        inp_tile[tile_y+ry, rx+sx] = inp[n, cur_in_channel, cur_y, cur_x]
                    else:
                        inp_tile[ry+tile_y, rx+sx] = 0


        ##### BOTTOM RIGHT
        if local_x < ker_w - 1 and local_y < ker_h - 1:

            var offset_x = stride_w * (TPB_X - ker_w) + ker_w
            var offset_y = stride_h * (TPB_Y-ker_h) + ker_h
            
            var gx_br = x_base + offset_x
            var gy_br = y_base + offset_y

            var tile_x =  sx + offset_x 
            var tile_y =  sy + offset_y


            for ry in range(stride_h):
                for rx in range(stride_w):
                    cur_x = gx_br + rx
                    cur_y = gy_br + ry

                    if cur_x< W and cur_y< H:
                        inp_tile[ry+tile_y, rx+tile_x] = \
                            inp[n, cur_in_channel, cur_y, cur_x]
                    else:
                        inp_tile[ry+tile_y, rx+tile_x] = 0

        barrier()

        # thread computes one pixel 
        if global_x < W_out and global_y < H_out:
            #var partial : output.element_type = 0
            var partial : Float64 = 0
            for y in range(ker_h):
                for x in range(ker_w):
                    # Again don't love this but hopefullly itll
                    # give more stability 
                    #partial += inp_tile[local_y*stride_h + y,
                    #                    local_x*stride_w + x] * ker_tile[y, x]
                    partial += Float64(inp_tile[local_y*stride_h + y,
                                        local_x*stride_w + x] * ker_tile[y, x])
            acc += partial

        barrier()

    # write out 
    if global_x < W_out and global_y < H_out:
        acc_new: Scalar[dtype] = Scalar[dtype](acc)
        simd: output.element_type = output.element_type(acc_new)
        # This is smelly, hard coding 32 for a sec
        #output[n, out_channel_idx, global_y, global_x] = Float32(acc)
        output[n, out_channel_idx, global_y, global_x] = simd
        #output[n, out_channel_idx, global_y, global_x] = acc


@compiler.register("conv2d_pad")
struct Conv2DCustomOp:

    @staticmethod
    fn execute[
        target : StaticString,
        batch: Int,
        in_channels: Int,
        out_channels: Int,
        H: Int,
        W: Int,
        ker_h: Int,
        ker_w: Int,
        stride_h: Int,
        stride_w: Int,
        pad_h: Int,
        pad_w:Int,
        dtype  : DType = DType.float32
    ](
        output : OutputTensor[dtype = dtype, rank = 4], # (batc, out_channels, h-ker_h+1, w-ker_w+1)
        input  : InputTensor [dtype = dtype, rank = 4],  # (batch, in_channels,  h, w)
        kernel : InputTensor [dtype = dtype, rank = 4], # (out_channels, in_channels,ker_h, ker w )
        ctx    : DeviceContextPtr,
    ) raises:
        
        # Calc total num elemenst to clear out buf
        alias H_out = (H + 2*pad_h - ker_h)//stride_h + 1
        alias W_out = (W + 2*pad_w - ker_w)//stride_w + 1
        alias out_elems = batch * out_channels * H_out * W_out

        # More housekeeping, layouts n' tensors
        var out_tensor  = output.to_layout_tensor()  
        var inp_tensor  = input.to_layout_tensor()
        var ker_tensor  = kernel.to_layout_tensor()
        alias in_layout   = inp_tensor.layout
        alias out_layout  = out_tensor.layout
        alias conv_layout = ker_tensor.layout


        # thread setups
        alias block_dim = THREADS_PER_BLOCK 

        alias grid_dim = (
            (W_out + TPB_X - 1) // TPB_X,# tiles in W
            (H_out + TPB_Y - 1) // TPB_Y,# tiles in H 
            batch * out_channels # fold batch & channel
        )


        gpu_ctx = ctx.get_device_context()
        gpu_ctx.enqueue_memset(
            DeviceBuffer[dtype](
                gpu_ctx,
                rebind[UnsafePointer[Scalar[dtype]]](out_tensor.ptr),
                out_elems,
                owning = False,
            ),
            0,
        )

        ctx.get_device_context().enqueue_function[
            conv2d_kernel[
                in_layout,
                out_layout,
                conv_layout,
                in_channels, 
                out_channels, 
                H, 
                W, 
                ker_h, 
                ker_w, 
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dtype
            ]
        ](
            out_tensor,
            inp_tensor,
            ker_tensor,
            grid_dim=grid_dim,      
            block_dim=block_dim,       
        )
