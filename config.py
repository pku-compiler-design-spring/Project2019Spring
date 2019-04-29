conv_shapes = [
    # batch_size, 
    # in_channel, 
    # inputs_height, 
    # inputs_width, 
    # out_channel, 
    # channel_per_group, 
    # kernel_height, 
    # kernel_width, 
    # if_bias=0, 
    # stride=1, 
    # padding=0, 
    # dilation=1, 
    # groups=1, 
    # dtype="float32"

    (1, 1024, 7, 7, 1024, 1024, 3, 3, 0, 1, 1, 1, 1), # yolo 24
    (1, 3, 448, 448, 64, 3, 7, 7, 0, 2, 3, 1, 1), # yolo 1
    (1, 512, 28, 28, 256, 128, 1, 1, 0, 1, 0, 1, 4), # yolo 7
    (1, 384, 27, 27, 64, 384, 1, 1, 1, 1, 0, 1, 1), # squeeze-net fire 8
    (1, 112, 14, 14, 224, 112, 3, 3, 0, 1, 1, 2, 1), # google-net inception4b-branch2b
    (8, 32, 7, 7, 128, 32, 5, 5, 0, 1, 2, 1, 1), # google-net inception5a-branch3b
]

gemm_shapes = [
    # batch
    # height
    # width
    # length

    (1, 1024, 1024, 1024),
    (2, 512, 512, 512),
    (3, 1024, 32, 1024),
    (1024, 32, 1024, 32),
    (4, 256, 1024, 128),
    (2, 4096, 200, 400),
]