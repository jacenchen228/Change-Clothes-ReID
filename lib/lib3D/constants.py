FOCAL_LENGTH = 500.
IMG_RES = 224   # size of the resized input img

# data augmentation for 3d
ROT_FACTOR = 30
NOISE_FACTOR = 0.4
SCALE_FACTOR = 0.25

# Permutation indices for the 14 ground truth joints (data format of the LSP dataset)
J14_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]
