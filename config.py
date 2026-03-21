VOCAB_SIZE = 8192          
D_MODEL = 896              
N_LAYERS = 12              
N_Q_HEADS = 14             
N_KV_HEADS = 2             
D_FF = 3584                
DROPOUT = 0.1
SEQ_LEN = 1024             
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 32      
LR = 6e-4                  
WEIGHT_DECAY = 0.1

MODEL_FOLDER = "checkpoints"
FINETUNE_MODEL_FOLDER = "checkpoints_finetune"

TOTAL_TRAINING_TOKENS = 2_800_000_000
WARMUP_STEPS = 1000

MAX_DATALOADER_WORKERS = 2
PREFETCH_FACTOR = 4
TOKENIZER_TEXT_BATCH_SIZE = 32
LOG_EVERY_STEPS = 8

ENABLE_TF32 = True
ENABLE_COMPILE = False
COMPILE_MODE = "max-autotune-no-cudagraphs"
