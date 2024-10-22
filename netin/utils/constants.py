import numpy as np

# GENERAL

EMPTY = [None, np.nan]

# GRAPHS

NO_HOMOPHILY = [0.5, None, np.nan]
NO_TRIADIC_CLOSURE = [0, None, np.nan]

CLASS_ATTRIBUTE = 'minority'

MAJORITY_LABEL = 'M'
MINORITY_LABEL = 'm'
CLASS_LABELS = [MAJORITY_LABEL, MINORITY_LABEL]

MAJORITY_VALUE = 0  # value for synthetic, and position for both synthetic and empirical pre-processed graphs
MINORITY_VALUE = 1  # value for synthetic, and position for both synthetic and empirical pre-processed graphs
CLASS_VALUES = [MAJORITY_VALUE, MINORITY_VALUE]

# GENERATIVE MODELS

EPSILON = 0.00001
MAX_TRIES_EDGE = 100

# DIRECTED_MODEL_NAME = 'Directed'
# UNDIRECTED_MODEL_NAME = 'Undirected'

# H_MODEL_NAME = 'Homophily'
# TC_MODEL_NAME = 'Triadic Closure'
#
# PA_MODEL_NAME = 'PA'
# PAH_MODEL_NAME = 'PAH'
# PATC_MODEL_NAME = 'PATC'
# PATCH_MODEL_NAME = 'PATCH'
#
# TCH_MODEL_NAME = 'TCH'
#
# DH_MODEL_NAME = 'DH'
# DPA_MODEL_NAME = 'DPA'
# DPAH_MODEL_NAME = 'DPAH'

# PA_MODEL_NAME = 'PA'
# PAH_MODEL_NAME = 'PAH'
# PATC_MODEL_NAME = 'PATC'
# PATCH_MODEL_NAME = 'PATCH'
# ERPATCH_MODEL_NAME = 'ERPATCH'

# TCH_MODEL_NAME = 'TCH'

# DH_MODEL_NAME = 'DH'
# DPA_MODEL_NAME = 'DPA'
# DPAH_MODEL_NAME = 'DPAH'

# HOMOPHILY_MODELS = [PAH_MODEL_NAME, PATCH_MODEL_NAME, DH_MODEL_NAME, DPAH_MODEL_NAME, H_MODEL_NAME]

# NODE METRICS

VALID_METRICS = ['degree', 'in_degree', 'out_degree', 'clustering',
                 'betweenness', 'closeness', 'eigenvector', 'pagerank']

# RANKING

RANK_RANGE = np.arange(0.1, 1 + 0.1, 0.1).astype(np.float32)
INEQUITY_BETA = 0.05
INEQUITY_OVER = 'over-represented'
INEQUITY_UNDER = 'under-represented'
INEQUITY_FAIR = 'fair'
INEQUALITY_CUTS = [0.3, 0.6]
INEQUALITY_HIGH = 'skewed'
INEQUALITY_MODERATE = 'moderate'
INEQUALITY_LOW = 'equality'
