from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "RSNA_Wavelet_Transformer" # Experiment name
_C.DEBUG = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.NUM_WORKERS = 1

_C.DIRS = CN()
_C.DIRS.DATA = "D:/Datasets/rsna/"
_C.DIRS.TRAIN = "proc/train/"
#_C.DIRS.VALID = "stage_1_test_images/"
_C.DIRS.TEST = "proc/test/"
_C.DIRS.TRAIN_CSV = "train_metadata.csv"
#_C.DIRS.VALID_CSV = "stage_1_test_metadata.csv"
_C.DIRS.TEST_CSV = "test_metadata.csv"
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"

_C.DATA = CN()
_C.DATA.CUTMIX = True
_C.DATA.MIXUP = False
_C.DATA.CM_ALPHA = 1.0
_C.DATA.MEAN = []
_C.DATA.STD = []
_C.DATA.IMG_SIZE = 224
_C.DATA.INP_CHANNEL = 3
_C.DATA.NUM_SLICES = 20

_C.FOLD = CN()
_C.FOLD.VALID = 0

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 5
_C.TRAIN.BATCH_SIZE = 8

_C.INFER = CN()
_C.INFER.TTA = False

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.SCHED = "cosine_warmup"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 4
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.EPS = 1e-3

_C.LOSS = CN()
_C.LOSS.WEIGHTS = [2., 1., 1., 1., 1., 1.]


_C.MODEL = CN()

_C.MODEL.WL_CHNS = [[12,32],48,192] # channels for the wavelet feature extractions
_C.MODEL.CONV_CHNS = [[64,128,256],[32,64,128,256],[48,64,128,256],[192,256]] # channels for the convolutional ops
_C.MODEL.LEVELS = [1,2,3,4] # decomposition levels for wavelet feature extraction

'''
_C.MODEL.ENCODER = CN()
#_C.MODEL.ENCODER.NAME = "se_resnext50_32x4d"

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.NAME = "lstm"
_C.MODEL.DECODER.NUM_LAYERS = 2
_C.MODEL.DECODER.IN_FEATURES = 2048
_C.MODEL.DECODER.HIDDEN_SIZE = 512
_C.MODEL.DECODER.BIDIRECT = True
_C.MODEL.DECODER.DROPOUT = 0.3
'''

_C.MODEL.NUM_CLASSES = 6

_C.CONST = CN()
_C.CONST.LABELS = [
  "any",
  "intraparenchymal", "intraventricular",
  "subarachnoid", "subdural", "epidural"
  ]