
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

import functools
partial = functools.partial



def num_tiles(size, blk_size):
    return 1 if blk_size<=0 else (size+blk_size-1)//blk_size

def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
  tiles, rem = divmod(x, tx)
  if rem:
    tiles += 1
  return tiles, rem

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
    ):
   
    m, k = lhs.shape
    k1, n = rhs.shape
    assert k == k1, f"k dimensions are not the same: {k} vs {k1}"

    #tile_m = group_metadata[1].shape[0]
    tk, k_rem = _calculate_irregular_num_tiles(k, blk_k)
    del tk
    
    tile_m = num_tiles(m, blk_m)
    tile_k = num_tiles(k, blk_k)
    tile_n = num_tiles(n, blk_n)

    print(f"lhs shape = {lhs.shape}, rhs shape = {rhs.shape}")
    print(f"tile_m, tile_k, tile_n = {tile_m}, {tile_k}, {tile_n}, K-rem={k_rem}")

    # kernel function
    def group_gmm_kernel(
          lhs_ref,
          rhs_ref,
          out_ref,
    ):
        n_i = pl.program_id(0)
        m_i = pl.program_id(1)
        k_i = pl.program_id(2)

        # initialize out_ref to 0: Note: initialization out_ref to 0 generates incorrect results          
        #@pl.when((m_i==0) & (n_i==0) & (k_i==0))
        #def _():
        #  out_ref[...] = jnp.zeros_like(out_ref)  
        
        input_dtype = lhs.dtype

        start_m = m_i*blk_m
        size_m  = jnp.minimum(blk_m, lhs.shape[0] - start_m).astype(jnp.int32).reshape(())
        start_k = k_i*blk_k
        size_k = jnp.minimum(blk_k, rhs.shape[0] - start_k).astype(jnp.int32).reshape(())
        start_n = n_i*blk_n
        size_n  = jnp.minimum(blk_n, rhs.shape[1] - start_n).astype(jnp.int32).reshape(())

         # Mask for valid rows/cols
        m_row_mask = (jnp.arange(blk_m) < size_m)[:, None]
        k_mask     = (jnp.arange(blk_k) < size_k)[:, None]
        n_col_mask = (jnp.arange(blk_n) < size_n)[None, :]
        
        def mask_block(x, *, dim):
            if k_rem == 0:
                return x

            orig_dtype = x.dtype
            iota = lax.broadcasted_iota(jnp.int32, x.shape, dim)
            x = x.astype(jnp.float32)
            return jnp.where(iota < k_rem, x, 0).astype(orig_dtype)

        def _mult(is_last_k_tile):
            if is_last_k_tile:
                mask_k_rem_lhs = partial(mask_block, dim=1)
                mask_k_rem_rhs = partial(mask_block, dim=int(transpose_rhs))
            else:
                mask_k_rem_lhs = lambda i: i
                mask_k_rem_rhs = lambda i: i

            if transpose_rhs:
                dot_general_dims = (((1,), (1,)), ((), ()))
            else:
                dot_general_dims = (((1,), (0,)), ((), ()))

            #x = pl.load(lhs_ref, (pl.ds(m_i*blk_m, blk_m), pl.ds(k_i*blk_k, blk_k)))
            #y = pl.load(rhs_ref, (pl.ds(k_i*blk_k, blk_k), pl.ds(n_i*blk_n, blk_n)))
            x = lhs_ref[pl.ds(start_m, blk_m) , pl.ds(start_k, blk_k)]
            y = rhs_ref[pl.ds(start_k, blk_k) , pl.ds(start_n, blk_n)]

            block_result = lax.dot_general(
                                        mask_k_rem_lhs(x).astype(input_dtype),
                                        mask_k_rem_rhs(y).astype(input_dtype),
                                        preferred_element_type=jnp.float32,
                                        dimension_numbers=dot_general_dims,
                                    )

            # Apply mask to block_result
            block_result = jnp.where(m_row_mask & n_col_mask, block_result, 0)

            # Accumulate across k dimension
            def store_block(val):
                # Always store the full block, but mask out-of-bounds elements
                pl.store(out_ref, (pl.ds(start_m, blk_m), pl.ds(start_n, blk_n)), val)

            _store_block = lambda: store_block(block_result)
            _acc_block   = lambda: store_block(pl.load(out_ref, (pl.ds(start_m, blk_m), pl.ds(start_n, blk_n))) + block_result)

            pl.when(k_i == 0)(_store_block)
            pl.when(k_i != 0)(_acc_block)


        lax.cond(k_i==tile_k-1, partial(_mult, True), partial(_mult, False))

          

    
    call_pallas_gmm_kernel = pl.pallas_call(
                              group_gmm_kernel,
                              out_shape=jax.ShapeDtypeStruct((m ,n), lhs.dtype),
                              grid=(tile_n, tile_m, tile_k),
                              interpret=interpret,
                              debug=False,  
                            )
    return call_pallas_gmm_kernel(lhs, rhs)
  



def main():

  M = 64
  K = 32
  N = 128

  lhs = jnp.full((M, K), 7.0, dtype=jnp.float32)
  rhs = jnp.full((K, N), 2.0, dtype=jnp.float32)
  
  pallas_out = gmm(lhs, rhs, transpose_rhs=False, blk_m=16, blk_k=16, blk_n=16, interpret=False)
  print(f"pallas_out={pallas_out}")      

  matmul_out = jnp.matmul(lhs, rhs)
  print(f"matmul_out={matmul_out}")      

  atol = 5e-1
  if not jnp.allclose(pallas_out,  matmul_out, atol):
      diff = jnp.abs(pallas_out - matmul_out).max()
      print(f"\nWARNING: pallas_out and matmul_out result mismatch; Max-diff={diff}\n")
  else:
      print("Test passed: pallas_out & matmul_out results matched\n")

 

if __name__ == "__main__":
    main()
