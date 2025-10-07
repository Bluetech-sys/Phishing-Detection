#-----------------
# Text transformer
#-----------------

MODEL_NAME = "microsoft/deberta-v3-base"
"""
"microsoft/deberta-v3-small"
"microsoft/deberta-v3-base"
"microsoft/deberta-v3-large"
"""
Classificaton_Hidden_Layers=[1024,512,128]

OUTPUT_DIR = "./deberta_phish_check"
TRAIN_CSV = "data/train.csv"        #Training Data
VALID_CSV = "data/valid.csv"        #Validate Reasults
NUM_LABELS = 2
MAX_LENGTH = 512   # tokenizer max tokens for model input
STRIDE = 128       # overlap between chunks (helps context)
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
SEED = 1025
AGGREGATION = "majority"  # "majority" or "mean" (probability average)
DROP_OUT_PROB=0.1