from jt_kernel.gmm import gmm as jt_gmm
#common func
from jt_kernel.common import (
    generate_inputs,
    num_of_cu,
    get_wrap_size,
    is_power_of_2,
    next_power_of_2,
    ragged_dot_reference,
    get_tiling,
    TRANS_LHS,
    TRANS_RHS,
    TRANS_OUT,
    TILING,
)