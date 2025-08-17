
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as tr

import functools
partial = functools.partial

def num_tiles(size, blk_size):
    return 1 if blk_size<=0 else (size+blk_size-1)//blk_size

@functools.partial(
    jax.jit,
    static_argnames=[
        "transpose_rhs",
        "blk_m",
        "blk_k",
        "blk_n",
        "interpret",
    ],
)
def gmm(
        lhs: jnp.ndarray,
        rhs: jnp.ndarray,
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
          out_ref,
    ) -> jnp.ndarray:
        n_i = pl.program_id(0)
        m_i = pl.program_id(1)
        k_i = pl.program_id(2)

        # initialize out_ref to 0: Note: initialization out_ref to 0 generates incorrect results          
        #@pl.when((m_i==0) & (n_i==0) & (k_i==0))
        #def _():
        #  out_ref[...] = jnp.zeros_like(out_ref)  
        
        n_i = pl.program_id(0)
        m_i = pl.program_id(1)
        k_i = pl.program_id(2)        

        start_m, start_k, start_n = m_i*blk_m, k_i*blk_k, n_i*blk_n

        x = lhs_ref[pl.ds(start_m, blk_m), pl.ds(start_k, blk_k)]
        y = rhs_ref[pl.ds(start_k, blk_k), pl.ds(start_n, blk_n)]
        
        dot_general_dims = (((1,), (1,)), ((), ())) if transpose_rhs else (((1,), (0,)), ((), ()))
                
        block_result = lax.dot_general(
                                        x.astype(x.dtype),
                                        y.astype(x.dtype),
                                        preferred_element_type=jnp.float32,
                                        dimension_numbers=dot_general_dims,
                                    )

        pl.atomic_add(out_ref, (pl.ds(start_m, blk_m), pl.ds(start_n, blk_n)) , block_result )

    call_pallas_gmm_kernel = pl.pallas_call(
                              group_gmm_kernel,
                              out_shape=jax.ShapeDtypeStruct((m ,n), lhs.dtype),
                              grid=(tile_n, tile_m, tile_k),
                              interpret=interpret,
                              debug=False,  
                            )
    return call_pallas_gmm_kernel(lhs, rhs)

def main():

    M, K , N = 1024, 2048, 512
    lhs = jnp.full((M, K), 3.0, dtype=jnp.float32)
    rhs = jnp.full((K, N), 2.0, dtype=jnp.float32)
    
    pallas_out = gmm(lhs, rhs,  blk_m=16, blk_k=16,blk_n=16, interpret=False)
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
