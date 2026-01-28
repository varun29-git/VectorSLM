# ============================================================
# MODEL SIZE (Scaled for A100 20GB ~400M Params)
# ============================================================

D_MODEL = 1024          
N_LAYERS = 24           
N_Q_HEADS = 16          
N_KV_HEADS = 8          # GQA (16 Q / 8 KV)
D_FF = 4096             # 4x D_MODEL
DROPOUT = 0.1           

# ============================================================
# SEQUENCE / BATCH 
# ============================================================

SEQ_LEN = 1024
BATCH_SIZE = 32         # Micro-batch (aggressive for A100)
GRAD_ACCUM_STEPS = 4    # Effective Batch = 128 (approx 131k tokens/step)

# ============================================================
# TRAINING SCHEDULE 
# ============================================================

EPOCHS = 1              # Controlled by train.py phases (Logical)
STEPS_PER_EPOCH = 2     # Ignored by token-budget loop

# ============================================================
# OPTIMIZATION
# ============================================================

LR = 3e-4               # Base LR (Controlled by train.py phases)
WEIGHT_DECAY = 0.1      

# ============================================================
# SYSTEM / INFRA
# ============================================================

MODEL_FOLDER = "checkpoints"
PRELOAD = None
VAL_SPLIT = 0.05
