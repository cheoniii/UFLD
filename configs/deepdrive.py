### Ultra-Fast-Lane-Detection Configurations
# TRAIN
num_epochs = 100
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

backbone = '18'
num_grid = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_folder = 'runs'
data_dir = '241221'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 2
cls_num_per_lane = 28
row_anchor = [
    64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112,
    116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
    168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
    220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
    272, 276, 280, 284
]

# input_size = (288, 800)
input_size = (64, 192)