
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as tr

from typing import Any

import functools
partial = functools.partial

GroupMetadata = Any

def num_tiles(size, blk_size):
    return 1 if blk_size<=0 else (size+blk_size-1)//blk_size


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: jnp.int32,
    block: jnp.int32
) -> GroupMetadata:
   
    print(f"group_sizes={group_sizes}, shape={group_sizes.shape}")
    split_sizes = []
    for size in group_sizes:
        while size > block:
            split_sizes.append(block)
            size -= block
        split_sizes.append(size)

    # 2) cumulative ends = where every tile finishes
    group_lengths = jnp.array(split_sizes, dtype=jnp.int32)
    group_ends=jnp.cumsum(group_lengths).astype(jnp.int32)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends]).astype(jnp.int32)
    print(f"group_offsets={group_offsets}, shape={group_offsets.shape}")
    print(f"group_lengths={group_lengths}, shape={group_lengths.shape}")

    return group_offsets, group_lengths, group_offsets.shape[0]


def gmm(
        lhs: jnp.ndarray,
        rhs: jnp.ndarray,
        group_sizes: jnp.ndarray,
        *,
        transpose_rhs : bool = False,
        blk_m: int = 32,
        blk_k: int = 32,
        blk_n: int = 32,
        interpret: bool = False,
    ) -> jnp.ndarray:

    group_offsets, group_lengths, num_groups = make_group_metadata(group_sizes=group_sizes, m=lhs.shape[0], block=blk_m)
    

    return _gmm(lhs,
         rhs,
         group_sizes,
         group_offsets,
         group_lengths,
         number_of_groups = num_groups,
         transpose_rhs=transpose_rhs,
         blk_m=blk_m,
         blk_k=blk_k,
         blk_n=blk_n,
         interpret=interpret,
        )



@functools.partial(
    jax.jit,
    static_argnames=[
        #"group_lengths",
        "number_of_groups",
        "transpose_rhs",
        "blk_m",
        "blk_k",
        "blk_n",
        "interpret",
    ],
)

def _gmm(
        lhs: jnp.ndarray,
        rhs: jnp.ndarray,
        group_sizes: jnp.ndarray,
        group_offsets: jnp.ndarray,
        group_lengths: jnp.ndarray,
        number_of_groups: int,
        *,
        transpose_rhs : bool = False,
        blk_m: int = 32,
        blk_k: int = 32,
        blk_n: int = 32,
        interpret: bool = False,
    ) -> jnp.ndarray:

    m, k = lhs.shape
    k1, n = rhs.shape
    assert k == k1, f"k dimensions are not the same: {k} vs {k1}"
    
    tile_m = num_tiles(m, blk_m)
    tile_k = num_tiles(k, blk_k)
    tile_n = num_tiles(n, blk_n)

    print(f"lhs shape = {lhs.shape}, rhs shape = {rhs.shape}")
    print(f"tile_m, tile_k, tile_n = {tile_m}, {tile_k}, {tile_n}")

    # kernel function
    def group_gmm_kernel(
          lhs_ref,
          rhs_ref,
          group_offsets,
          group_lengths,
          out_ref,
    ) -> jnp.ndarray:
       
        # initialize out_ref to 0: Note: initialization out_ref to 0 generates incorrect results          
        #@pl.when((m_i==0) & (n_i==0) & (k_i==0))
        #def _():
        #  out_ref[...] = jnp.zeros_like(out_ref)  
        
        n_i = pl.program_id(0)
        m_i = pl.program_id(1)
        k_i = pl.program_id(2)        

        start_m, start_k, start_n = group_offsets[m_i], k_i*blk_k, n_i*blk_n
        size_m = blk_m #group_lengths[m_i]

        def mask_group(x, group_size, *, dim):
            orig_dtype = x.dtype
            iota = lax.broadcasted_iota(jnp.int32, (group_size, blk_k), dim)
            x = x.astype(jnp.float32)
            return jnp.where(iota < group_size, x, 0).astype(orig_dtype)



        x = lhs_ref[pl.ds(start_m, size_m), pl.ds(start_k, blk_k)]
        y = rhs_ref[pl.ds(start_k, blk_k), pl.ds(start_n, blk_n)]
        
        dot_general_dims = (((1,), (1,)), ((), ())) if transpose_rhs else (((1,), (0,)), ((), ()))
                
        block_result = lax.dot_general(
                                        x.astype(x.dtype),
                                        y.astype(x.dtype),
                                        preferred_element_type=jnp.float32,
                                        dimension_numbers=dot_general_dims,
                                    )

        pl.atomic_add(out_ref, (pl.ds(start_m, size_m), pl.ds(start_n, blk_n)) , block_result )

    call_pallas_gmm_kernel = pl.pallas_call(
                              group_gmm_kernel,
                              out_shape=jax.ShapeDtypeStruct((m ,n), lhs.dtype),
                              grid=(tile_n, number_of_groups, tile_k),
                              interpret=interpret,
                              debug=False,  
                            )
    return call_pallas_gmm_kernel(lhs, rhs, group_offsets, group_lengths)

def main():

    M, K , N = 64, 256, 512
    lhs = jnp.full((M, K), 3.0, dtype=jnp.float32)
    rhs = jnp.full((K, N), 2.0, dtype=jnp.float32)

    group_sizes = jnp.array([16, 16, 32], dtype=jnp.int32)
    
    pallas_out = gmm(lhs, rhs,  group_sizes, blk_m=16, blk_k=16,blk_n=16, interpret=False)
    pallas_out.block_until_ready()
    print(f"pallas_out={pallas_out}")      

    matmul_out = jnp.matmul(lhs, rhs)
    print(f"matmul_out={matmul_out}")

    atol = 5e-1
    if not jnp.allclose(pallas_out,  matmul_out, atol):
        diff = jnp.abs(pallas_out - matmul_out).max()
        print(f"\n\tWARNING: pallas_out and matmul_out result mismatch; Max-diff={diff}\n")
    else:
        print("\n\tTest passed: pallas_out & matmul_out results matched\n")      

 

if __name__ == "__main__":
    main()
