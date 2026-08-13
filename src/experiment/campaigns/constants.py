"""Paper-locked empirical grids and task identifiers."""

L_GRID = (0, 1, 2, 4, 8, 16, 32, 64)
TAU_PLUS_GRID = (2, 4, 8, 16, 32, 64)
R_GRID = (1, 2, 4, 8, 16, 32, 64)
H_GRID = (1, 2, 4, 8)
D_GRID = (1, 2, 4, 8, 16, 32, 64)
N_WIDTH_GRID = tuple(range(3, 13))
P_GRID = (1, 2, 4, 8, 16, 32)
ARITY_GRID = (1, 2, 4, 8, 16)
M_GRID = tuple(2**power for power in range(14))
WINDOW_GRID = tuple(2**power for power in range(10))
GAP_GRID = (0, 1, 2, 4, 8, 16, 32, 64)
SAMPLE_GRID = (128, 256, 512, 1024, 2048, 4096)
POOL_GRID = tuple(2**power for power in range(9))
DEPENDENCE_GRID = (1, 2, 4, 8, 16)

SHOT_ARCHITECTURES = (
    (4, 1),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 8),
    (8, 16),
    (16, 16),
    (16, 32),
    (16, 64),
)

ATLAS_TASKS = (
    tuple(f"F_mem_{value}" for value in L_GRID)
    + tuple(f"F_multi_{value}" for value in H_GRID)
    + tuple(f"F_sp_{value}" for value in D_GRID)
    + tuple(f"F_deg_{value}" for value in P_GRID)
    + tuple(f"F_int_{value}" for value in ARITY_GRID)
    + tuple(f"F_C{value}" for value in range(1, 7))
)

ARCHITECTURE_TIERS = {
    "small": (4, 4, 8),
    "medium": (8, 16, 32),
    "large": (16, 64, 64),
}

