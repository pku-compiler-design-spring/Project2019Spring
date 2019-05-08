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
    (8, 384, 27, 27, 64, 384, 1, 1, 1, 1, 0, 1, 1), # squeeze-net fire 8
    (4, 112, 14, 14, 224, 112, 3, 3, 0, 1, 1, 2, 1), # google-net inception4b-branch2b
]

gemm_shapes = [
    # batch
    # height
    # width
    # length

    (1, 1024, 1024, 1024),
    (2, 512, 512, 512),
    (8, 1024, 32, 1024),
]