import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from jax._src.lax.control_flow.for_loop import for_loop
from jax.experimental import pallas as pl

import functools



def _calculate_num_blocks(size, blk_size):
  return pl.cdiv(size, blk_size)

def _align_dims_to_blocks(size, blk_size):
  return pl.cdiv(size, blk_size)*blk_size - size
    

@functools.partial(jax.jit, static_argnames=['blk_r', 'blk_c', 'pad_val', ])
def padding(A, *, blk_r: jnp.int32=16 , blk_c: jnp.int32=16, pad_val: jnp.int32=0):
   
    rows, cols = A.shape  
    pad_top, pad_left = 0, 0
    pad_bottom, pad_right = _align_dims_to_blocks(rows, blk_r), _align_dims_to_blocks(cols, blk_c)
    return jnp.pad(A, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=pad_val )


############################################################################

@functools.partial(
                  jax.jit, 
                  static_argnames=[
                                  "m_blocksize",
                                  "k_blocksize",
                                  "n_blocksize",
                                  "interpret",
                                  "debug"]
                  )
def _matmul_(x, y, *, m_blocksize, k_blocksize, n_blocksize, interpret=False, debug=False):

    
  m1, n1, k1 = x.shape[0], y.shape[1], x.shape[1]

  print(f"x shape = {x.shape}, y shape = {y.shape}")

  lhs=padding(x, blk_r=m_blocksize , blk_c=k_blocksize, pad_val=0)
  rhs=padding(y, blk_r=k_blocksize , blk_c=n_blocksize, pad_val=0)

  M, N, K = lhs.shape[0], rhs.shape[1], rhs.shape[1]

  m_blocks = _calculate_num_blocks(M, m_blocksize)
  k_blocks = _calculate_num_blocks(K, k_blocksize)
  n_blocks = _calculate_num_blocks(N, n_blocksize)

  print(f"lhs shape = {lhs.shape}, rhs shape = {rhs.shape}")
  print(f"m_blocks, k_blocks, n_blocks = {m_blocks}, {k_blocks}, {n_blocks}")

 
  #########   pallas kernel ##################
  def matmul_kernel(x_ref, y_ref, o_ref):
    
    acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)

    def _mult(kk, acc_ref):
      x_block = pl.load(x_ref, (slice(None), pl.ds(kk * k_blocksize, k_blocksize)))
      y_block = pl.load(y_ref, (pl.ds(kk * k_blocksize, k_blocksize), slice(None)))
      acc_ref[:, :] += pl.dot(x_block, y_block)

    acc = for_loop(k_blocks, _mult, acc).astype(o_ref.dtype)
    o_ref[:, :] = acc
    ################################

  kernel_call = pl.pallas_call(
                        matmul_kernel,
                        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
                        interpret=interpret,
                        debug=debug,
                        in_specs=[
                            pl.BlockSpec((m_blocksize, K), lambda i, _: (i, 0)),
                            pl.BlockSpec((K, n_blocksize), lambda _, j: (0, j)),
                        ],
                        out_specs=pl.BlockSpec((m_blocksize, n_blocksize), lambda i, j: (i, j)),
                        grid=(m_blocks, n_blocks),
    )

  out_full = kernel_call(x, y)
  return out_full[:m1, :n1]


def main():  
  m, k, n = 512, 1010, 2048

  lhs = jnp.full((m, k), 10.0)
  rhs = jnp.full((k, n), 5.0)
  pallas_out = _matmul_(lhs, rhs, m_blocksize=16, k_blocksize=16, n_blocksize=16, interpret=False, debug=False)
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
