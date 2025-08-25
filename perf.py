
import jax
import jax.numpy as jnp
import megablox_new.gmm as mblx_atomic_add
import megablox.gmm as mblx_reduction_dim_loop
from megablox.gmm import tgmm as tgmm_loop


import functools
import timeit



TILING: tuple[int, int, int] = (64, 64, 64)
# Default transposition.
TRANS_LHS: bool = False
TRANS_RHS: bool = False
TRANS_OUT: bool = False


def generate_inputs(
    M: int,
    K: int,
    N: int,
    G: int,
    preferred_element_type: jnp.dtype = jnp.bfloat16,
    trans_lhs: bool = TRANS_LHS,
    trans_rhs: bool = TRANS_RHS,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    assert M > 0, f"Number of lhs rows M must be positive (M = {M})."
    assert K > 0, f"Number of lhs columns / rhs rows K must be positive (K = {K})."
    assert N > 0, f"Number of rhs columns N must be positive (N = {N})."
    assert G > 0, f"Number of groups G must be positive (G = {G})."
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    lhs_row, lhs_col = (K, M) if trans_lhs else (M, K)
    rhs_row, rhs_col = (N, K) if trans_rhs else (K, N)
    lhs = jax.random.normal(k1, (lhs_row, lhs_col), dtype=preferred_element_type)
    rhs = jax.random.normal(k2, (G, rhs_row, rhs_col), dtype=preferred_element_type)   
    return lhs, rhs


def matmul_flops(m: int, k: int, n: int):
  return 2 * m * k * n


def _perf_(func_, ntrials, m, k, n, str ):

  func_().block_until_ready() #warm-up
  time = timeit.timeit(lambda: func_().block_until_ready(), number=ntrials)
  avg_time =  (time/ntrials)
  _ops = matmul_flops(m,k,n)
  print(f"{str}: Average time in {ntrials} runs: {avg_time:.6f} sec,  TFLOP/s = {_ops/(avg_time/1e3)/1e12:.6f}")


def main(unused_argv):
   
    preferred_element_type=jnp.bfloat16 #jnp.float32
    precision_ = "F32_F32_F32" if preferred_element_type==jnp.float32 else "BF16_BF16_F32"
    print(f"precision_ = {precision_}")
    
    M, K, N = 393216, 2048, 1408
    
    group_sizes = jnp.array([49152,65536,32768,34816,63488,49152,32768,65536], dtype=jnp.int32)    
    G = group_sizes.shape[0]

    lhs, rhs = generate_inputs(M, K, N, G,  preferred_element_type=preferred_element_type, trans_lhs=False, trans_rhs=False)        
    
    print(f"lhs shape={lhs.shape}, rhs shape={rhs.shape}, group_sizes={group_sizes} preferred_element_type={preferred_element_type}")

    #mblx_atomic_add_gmm = functools.partial(mblx_atomic_add.gmm, lhs, rhs, group_sizes=group_sizes, tiling=TILING, interpret=False)
    #mblx_reduction_loop = functools.partial(mblx_reduction_dim_loop.gmm, lhs, rhs, group_sizes=group_sizes, tiling=TILING, interpret=False)
    #jax_ragged_dot = functools.partial(jax.lax.ragged_dot, lhs, rhs, group_sizes=group_sizes,precision=precision_, )
    
    #mblx_atomic_add_tgmm = functools.partial(mblx_atomic_add.tgmm, lhs, rhs, group_sizes=group_sizes, tiling=TILING, interpret=False)
    mblx_reduction_loop_tgmm = functools.partial(tgmm_loop, lhs, rhs[0], group_sizes=group_sizes, tiling=TILING, interpret=False) 
    
    
    ntrials=5
    #_perf_(mblx_atomic_add_gmm, ntrials, M, K, N, "megablox-gmm-atomic_add")
    #_perf_(mblx_reduction_loop, ntrials, M, K, N, "megablox-gmm-reduction-index-loop")
    #_perf_(jax_ragged_dot, ntrials, M, K, N, "ragged_dot")
    #_perf_(mblx_atomic_add_tgmm, ntrials, M, K, N, "megablox-tgmm-atomic_add")
    _perf_(mblx_reduction_loop_tgmm, ntrials, M, K, N, "megablox-gmm-reduction-dimloop")
  
    
if __name__ == "__main__":
    from absl import app
    app.run(main)
