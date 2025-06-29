from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from testing import assert_equal

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer
from tensor import InputTensor, OutputTensor

alias TPB = 32
alias SIZE = 2
alias BATCH_SIZE = 2
alias BLOCKS_PER_GRID = (1,1)
alias THREADS_PER_BLOCK = (TPB,TPB)
alias dtype = DType.float32

alias TILE_M = TPB
alias TILE_N = TPB
alias TILE_K = TPB

alias A_ROW = SIZE
alias A_COL_B_ROW = SIZE
alias B_COL = SIZE

alias a_layout = Layout.row_major(BATCH_SIZE, A_ROW, A_COL_B_ROW)
alias b_layout = Layout.row_major(BATCH_SIZE, A_COL_B_ROW, B_COL)
alias out_layout = Layout.row_major(BATCH_SIZE, A_ROW, B_COL)


fn bmm_tiled[a_layout:Layout, b_layout:Layout, out_layout:Layout](
    A:   LayoutTensor[mut=False, dtype, a_layout],    # inp 1 BxMxN
    B:   LayoutTensor[mut=False, dtype, b_layout],    # inp 1 BxNxK
    output: LayoutTensor[mut=False, dtype, out_layout],    # inp 1 BxMxK
    a_row_size: Int,
    a_col_b_row_size: Int,
    b_col_size: Int
):

    # Need the batch idx
    var batch_id = block_idx.x

    # Coords for the maxtirx
    var by = block_idx.y
    var bx = block_idx.z

    # Tile wise i
    var row = by * TILE_M + thread_idx.y
    var col = bx * TILE_N + thread_idx.x

    # Shared mem
    var a_shared = tb[dtype]().row_major[TILE_M, TILE_K]().shared().alloc()
    var b_shared = tb[dtype]().row_major[TILE_K, TILE_N]().shared().alloc()

    # Accumulated val
    var acc: output.element_type = 0.0


    # Loop of K (a_col_b_row)
    for ko in range(0, a_col_b_row_size, TILE_K):

        # Loop tile A, A is NxK, so a_row_size x a_col_b_row_size.. 
        # so guard on the row, is easy, the col is ko + cur_thread idx
        if row < a_row_size and ko + thread_idx.x < a_col_b_row_size:
            a_shared[thread_idx.y, thread_idx.x] = A[batch_id, row, 
                                    ko + thread_idx.x]
        else:
            a_shared[thread_idx.y, thread_idx.x ] = 0.0

        # Load tile of B  - gaurds are some idea as a but for KxM
        if ko + thread_idx.y < a_col_b_row_size and col < b_col_size:
            b_shared[thread_idx.y, thread_idx.x] = B[batch_id, ko + thread_idx.y, col]
        else:
            b_shared[thread_idx.y, thread_idx.x] = 0.0

        # Let tiles load
        barrier()

        # accum across shared tile
        for k in range(TILE_K):
            acc += a_shared[thread_idx.y, k] * b_shared[k, thread_idx.x]

        barrier()

    # Each batch, row, col will write
    #if row < a_row_size and col < a_col_b_row_size:
    if row < a_row_size and col < b_col_size:
        output[batch_id, row, col] = acc
    return


@compiler.register("bmm_tiled")
struct BatchedMatMul:

    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor[rank=3], #(batch, row, col)
        A: InputTensor[dtype = output.dtype, rank = output.rank],
        B: InputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Most of this is similar to gpu puzzles, 
         #house keeping for tensors and layouts 
        out_tensor = output.to_layout_tensor()
        a_tensor = A.to_layout_tensor()
        b_tensor = B.to_layout_tensor()

        # Prepended these with cur to not clobber global vars... but would 
        # that even be a problem I never tried 
        alias cur_a_layout = a_tensor.layout
        alias cur_b_layout = b_tensor.layout
        alias cur_out_layout = out_tensor.layout

        #TODO
        #if target == "cpu":
        #    _bmm_cpu(C, A, B)

        @parameter
        if target == "gpu":

            gpu_ctx = ctx.get_device_context()

            var batch_size = output.dim_size(0)
            var M      = output.dim_size(1)
            var N      = output.dim_size(2)
            var K      = A.dim_size(2)

            grid_dim = (
                batch_size,
                (M + TILE_M - 1) // TILE_M,   # row tiles
                (N + TILE_N - 1) // TILE_N    # col tiles
            )

            
            gpu_ctx.enqueue_function[bmm_tiled[cur_a_layout, cur_b_layout, cur_out_layout]](
                a_tensor, 
                b_tensor, 
                out_tensor, 
                M,
                K,
                N,
                grid_dim = grid_dim,
                block_dim = THREADS_PER_BLOCK
            )
        else:
            raise Error("Unsupported target " + target)

