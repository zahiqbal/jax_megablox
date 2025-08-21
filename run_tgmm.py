import jax
import jax.numpy as jnp

#from jax.jax.experimental.pallas.ops.gpu.megablox import gmm import gmm
from megablox_new.gmm import gmm
from megablox_new.gmm import tgmm
import sys

INTERPRET=False


# ====================================================================

def verify(mat1, mat2, message="", atol=5e-1):
    print(f" \n\t mat1 shape = {mat1.shape}\n\t mat2 shape  = {mat2.shape}\n")
    if not jnp.allclose(mat1, mat2, atol=atol):
        diff = jnp.abs(out_gmm - out_ragged).max()
        raise ValueError(
            f"\n\tMismatch between mat1 and mat2. Max diff={diff}"
            f" \n\t mat1 shape = {mat1.shape}\n\t mat2 shape  = {mat2.shape}\n"
            f"out_gmm = {mat1}\n\nout_ragged = {mat2} "
        )
    else:
        print(f"\n\tTest Passed:  mat1 and mat2 result matched...\n")


def check_gmm_vs_ragged_dot(lhs, rhs, group_sizes, tiling, atol=5e-1):
    out_gmm = gmm(lhs, rhs, group_sizes=group_sizes, tiling=tiling, interpret=INTERPRET)
       
    out_ragged = jax.lax.ragged_dot(lhs, rhs, group_sizes=group_sizes)
    
    if not jnp.allclose(out_gmm, out_ragged, atol=atol):
        diff = jnp.abs(out_gmm - out_ragged).max()
        raise ValueError(
            f"\n\tMismatch between gmm and ragged_dot. Max diff={diff}"
            f" \n\tmegablox out shape = {out_gmm.shape}\n\t out_ragged shape  = {out_ragged.shape}\n"
            f"out_gmm = {out_gmm}\n\nout_ragged = {out_ragged} "
        )
    else:
        print(f"\n\tTest Passed:  gmm and ragged_dot result matched...\n")

    
def test_gpu_megablox():

    key = jax.random.PRNGKey(0)


    M, K , N = 64, 256, 512
    group_sizes = jnp.array([3, 4, 6, 8, 11, 32], dtype=jnp.int32)
    g = group_sizes.shape[0]
    lhs = jnp.full((K, M), 3.0, dtype=jnp.float32)
    rhs = jnp.full((M, N), 2.0, dtype=jnp.float32)
    tiling = (16, 16, 16)
    #out_gmm = gmm(lhs, rhs, group_sizes=group_sizes, tiling=tiling, interpret=INTERPRET)
    out_tgmm = tgmm(lhs, rhs, group_sizes=group_sizes, tiling=tiling, interpret=INTERPRET)
    print(f"\n out_tgmm shape = {out_tgmm.shape}\n\t out_tgmm = {out_tgmm} ")

    print(f"\n\n\tlhs@rhs = {lhs@rhs} ")

    #verify(rhs, out_tgmm, "Za", )  


    

    # ---------- Test 1 ----------
    """M, K, N = 512, 512, 256
    group_sizes = jnp.array([32, 64, 128, 32, 64, 128, 64], dtype=jnp.int32)
    g = group_sizes.shape[0]
    lhs = jax.random.normal(key, (M, K))
    rhs = jax.random.normal(key, (g, K, N))
    tiling = (16, 16, 16)
    # This part of code is for checking gmm vs ragged dot accuracy
    check_gmm_vs_ragged_dot(lhs, rhs, group_sizes, tiling)
    

    # ---------- Test 2 ----------
    
    M, K, N, E = 512, 2048, 1024, 160
    lhs = jax.random.normal(key, (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(key, (E, K, N), dtype=jnp.float32)

    base = jnp.full((E,), 3, dtype=jnp.int32)   # each expert gets 3 tokens
    extra = jnp.arange(E) < 32                  # first 32 experts get 1 more
    group_sizes = base + extra.astype(jnp.int32)  # sum=160*3+32=512

    #print(f"group_sizes 1: {group_sizes}")

    tile_size = (16, 16, 16)  # no partial leftover
   
    check_gmm_vs_ragged_dot(lhs, rhs, group_sizes, tile_size, 5e-1)
    print("Test 3 => PASSED")    # <= DID NOT PASS!!"""
   


if __name__ == "__main__":
    test_gpu_megablox()
