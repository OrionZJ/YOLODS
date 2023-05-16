import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
#sys.path.append("lib/models")
#sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, C2f, SPPF
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized

"""
MCnet_SPP = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
# [ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ [17, 20, 23], Detect,  [13, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ 17, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, SPP, [8, 2, [5, 9, 13]]] #segmentation output
]
# [2,6,3,9,5,13], [7,19,11,26,17,39], [28,64,44,103,61,183]

MCnet_0 = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [8, 2, 3, 1]], #Driving area segmentation output

[ 16, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [8, 2, 3, 1]], #Lane line segmentation output
]


# The lane line and the driving area segment branches share information with each other
MCnet_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck

[ 16, Conv, [256, 64, 3, 1]],   #33
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ [-1,2], Concat, [1]], #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39   
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40 lane line segment neck

[ [31,39], Concat, [1]],    #41
[ -1, Conv, [32, 8, 3, 1]],     #42    Share_Block


[ [32,42], Concat, [1]],     #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [16, 2, 3, 1]], #45 Driving area segmentation output


[ [40,42], Concat, [1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [16, 2, 3, 1]] #48Lane line segmentation output
]

# The lane line and the driving area segment branches without share information with each other
MCnet_no_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 3, 3, 1]], #34 Driving area segmentation output

[ 16, Conv, [256, 64, 3, 1]],   #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ [-1,2], Concat, [1]], #37
[ -1, BottleneckCSP, [128, 64, 1, False]],  #38
[ -1, Conv, [64, 32, 3, 1]],    #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, Conv, [32, 16, 3, 1]],    #41
[ -1, BottleneckCSP, [16, 8, 1, False]],    #42 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [8, 2, 3, 1]] #44 Lane line segmentation output
]

MCnet_feedback = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 2, 3, 1]], #34 Driving area segmentation output

[ 16, Conv, [256, 128, 3, 1]],   #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ -1, BottleneckCSP, [128, 64, 1, False]],  #38
[ -1, Conv, [64, 32, 3, 1]],    #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, Conv, [32, 16, 3, 1]],    #41
[ -1, BottleneckCSP, [16, 8, 1, False]],    #42 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [8, 2, 3, 1]] #44 Lane line segmentation output
]


MCnet_Da_feedback1 = [
[46, 26, 35],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16     backbone+fpn
[ -1,Conv,[256,256,1,1]],   #17


[ 16, Conv, [256, 128, 3, 1]],   #18
[ -1, Upsample, [None, 2, 'nearest']],  #19
[ -1, BottleneckCSP, [128, 64, 1, False]],  #20
[ -1, Conv, [64, 32, 3, 1]],    #21
[ -1, Upsample, [None, 2, 'nearest']],  #22
[ -1, Conv, [32, 16, 3, 1]],    #23
[ -1, BottleneckCSP, [16, 8, 1, False]],    #24 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #25
[ -1, Conv, [8, 2, 3, 1]], #26 Driving area segmentation output


[ 16, Conv, [256, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35Lane line segmentation output


[ 23, Conv, [16, 16, 3, 2]],     #36
[ -1, Conv, [16, 32, 3, 2]],    #2 times 2xdownsample    37

[ [-1,17], Concat, [1]],       #38
[ -1, BottleneckCSP, [288, 128, 1, False]],    #39
[ -1, Conv, [128, 128, 3, 2]],      #40
[ [-1, 14], Concat, [1]],       #41
[ -1, BottleneckCSP, [256, 256, 1, False]],     #42
[ -1, Conv, [256, 256, 3, 2]],      #43
[ [-1, 10], Concat, [1]],   #44
[ -1, BottleneckCSP, [512, 512, 1, False]],     #45
[ [39, 42, 45], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]] #Detect output 46
]



# The lane line and the driving area segment branches share information with each other and feedback to det_head
MCnet_Da_feedback2 = [
[47, 26, 35],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[25, 28, 31, 33],   #layer in Da_branch to do SAD
[34, 37, 40, 42],   #layer in LL_branch to do SAD
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16     backbone+fpn
[ -1,Conv,[256,256,1,1]],   #17


[ 16, Conv, [256, 128, 3, 1]],   #18
[ -1, Upsample, [None, 2, 'nearest']],  #19
[ -1, BottleneckCSP, [128, 64, 1, False]],  #20
[ -1, Conv, [64, 32, 3, 1]],    #21
[ -1, Upsample, [None, 2, 'nearest']],  #22
[ -1, Conv, [32, 16, 3, 1]],    #23
[ -1, BottleneckCSP, [16, 8, 1, False]],    #24 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #25
[ -1, Conv, [8, 2, 3, 1]], #26 Driving area segmentation output


[ 16, Conv, [256, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35Lane line segmentation output


[ 23, Conv, [16, 64, 3, 2]],     #36
[ -1, Conv, [64, 256, 3, 2]],    #2 times 2xdownsample    37

[ [-1,17], Concat, [1]],       #38

[-1, Conv, [512, 256, 3, 1]],     #39
[ -1, BottleneckCSP, [256, 128, 1, False]],    #40
[ -1, Conv, [128, 128, 3, 2]],      #41
[ [-1, 14], Concat, [1]],       #42
[ -1, BottleneckCSP, [256, 256, 1, False]],     #43
[ -1, Conv, [256, 256, 3, 2]],      #44
[ [-1, 10], Concat, [1]],   #45
[ -1, BottleneckCSP, [512, 512, 1, False]],     #46
[ [40, 42, 45], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]] #Detect output 47
]

MCnet_share1 = [
[24, 33, 45],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[25, 28, 31, 33],   #layer in Da_branch to do SAD
[34, 37, 40, 42],   #layer in LL_branch to do SAD
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ -1, Conv, [64, 32, 3, 1]],    #28
[ -1, Upsample, [None, 2, 'nearest']],  #29
[ -1, Conv, [32, 16, 3, 1]],    #30

[ -1, BottleneckCSP, [16, 8, 1, False]],    #31 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #32
[ -1, Conv, [8, 2, 3, 1]], #33 Driving area segmentation output

[ 16, Conv, [256, 128, 3, 1]],   #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39

[ 30, SharpenConv, [16,16, 3, 1]], #40
[ -1, Conv, [16, 16, 3, 1]], #41
[ [-1, 39], Concat, [1]],   #42
[ -1, BottleneckCSP, [32, 8, 1, False]],    #43 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [8, 2, 3, 1]] #45 Lane line segmentation output
]"""


# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
[24, 33, 42],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15  1/4
[ [-1,4], Concat, [1]],     #16         #Encoder

[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18  1/8
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21  1/16
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

[ 16, Conv, [256, 128, 3, 1]],   #25  1/4 - 2
[ -1, Upsample, [None, 2, 'nearest']],  #26  1/2 -4
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ -1, Conv, [64, 32, 3, 1]],    #28  1/2 - 6
[ -1, Upsample, [None, 2, 'nearest']],  #29  1 - 12
[ -1, Conv, [32, 16, 3, 1]],    #30  1 - 14
[ -1, BottleneckCSP, [16, 8, 1, False]],    #31
[ -1, Upsample, [None, 2, 'nearest']],  #32  2 - 28
[ -1, Conv, [8, 2, 3, 1]], #33 2 - 30
    # Driving area segmentation head

[ 16, Conv, [256, 128, 3, 1]],   #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40
[ -1, Upsample, [None, 2, 'nearest']],  #41
[ -1, Conv, [8, 2, 3, 1]] #42 Lane line segmentation head
]

# The lane line and the driving area segment branches without share information with each other and without link
YOLODSn = [
    [45, 65, 85],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],    # 0 p1
    [-1, Conv, [32, 64, 3, 2]],     # 1  1/2
    [-1, Conv, [64, 128, 1, 1]],    # 2 p2
    [-1, C2f, [128, 128, True]],     # 3
    [-1, C2f, [128, 128, True]],     # 4
    [-1, C2f, [128, 128, True]],     # 5
    [-1, Conv, [128, 256, 3, 2]],    # 6 p3  1/4
    [-1, C2f, [256, 256, True]],     # 7
    [-1, C2f, [256, 256, True]],     # 8
    [-1, C2f, [256, 256, True]],     # 9
    [-1, C2f, [256, 256, True]],     # 10
    [-1, C2f, [256, 256, True]],     # 11
    [-1, C2f, [256, 256, True]],     # 12
    [-1, Conv, [256, 512, 3, 2]],    # 13 p4  1/8
    [-1, C2f, [512, 512, True]],     # 14
    [-1, C2f, [512, 512, True]],     # 15
    [-1, C2f, [512, 512, True]],     # 16
    [-1, C2f, [512, 512, True]],     # 17
    [-1, C2f, [512, 512, True]],     # 18
    [-1, C2f, [512, 512, True]],     # 19
    [-1, Conv, [512, 1024, 3, 2]],   # 20 p5  1/16
    [-1, C2f, [1024, 1024, True]],    # 21
    [-1, C2f, [1024, 1024, True]],    # 22
    [-1, C2f, [1024, 1024, True]],    # 23
    [-1, SPPF, [1024, 1024, 5]],      # 24

    [-1, nn.Upsample, [None, 2, 'nearest']],      #25  1/8
    [[-1, 19], Concat, [1]],        #26  # cat backbone P4
    [-1, C2f, [1536, 512]],       # 27
    [-1, C2f, [512, 512]],       # 28
    [-1, C2f, [512, 512]],       # 29

    [-1, nn.Upsample, [None, 2, 'nearest']],        # 30  1/4
    [[-1, 12], Concat, [1]],     # 31 # cat backbone P3
    [-1, C2f, [768, 256]],       # 32
    [-1, C2f, [256, 256]],       # 33
    [-1, C2f, [256, 256]],       # 34

    [-1, Conv, [256, 256, 3, 2]],        # 35  1/8
    [[-1, 29], Concat, [1]],        # 36
    [-1, C2f, [768, 512]],       # 37
    [-1, C2f, [512, 512]],       # 38
    [-1, C2f, [512, 512]],       # 39

    [-1, Conv, [512, 512, 3, 2]],        # 40  1/16
    [[-1, 24], Concat, [1]],        # 41  # cat head P5
    [-1, C2f, [1536, 1024]],      # 42
    [-1, C2f, [1024, 1024]],      # 43
    [-1, C2f, [1024, 1024]],      # 44

    [ [34, 39, 44], Detect,
      [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]],
       [256, 512, 1024]]], #Detection head 45

    # seg drive area
    [24, nn.Upsample, [None, 2, 'nearest']],  # 46  1/8
    [[-1, 19], Concat, [1]],  # 47  # cat backbone P4
    [-1, C2f, [1536, 512]],  # 48
    [-1, C2f, [512, 512]],  # 49
    [-1, C2f, [512, 512]],  # 50

    [-1, nn.Upsample, [None, 2, 'nearest']],  # 51  1/4
    [[-1, 12], Concat, [1]],  # 52 # cat backbone P3
    [-1, Conv, [768, 256, 3, 1]],   # 53 1/4 - 2

    [-1, C2f, [256, 256]],  # 54
    [-1, nn.Upsample, [None, 2, 'nearest']],    # 55 1/2 - 4
    [-1, Conv, [256, 256, 3, 1]],   # 56  1/2 - 6

    [-1, C2f, [256, 256]],  # 57
    [-1, Upsample, [None, 2, 'nearest']],  # 58  1 - 12
    [-1, Conv, [256, 256, 3, 1]],   # 59 1 - 14
    [-1, C2f, [256, 256]],  # 60

    [-1, Upsample, [None, 2, 'nearest']],  # 61  2 - 28
    [-1, Conv, [256, 256, 3, 1]],  # 62 2 - 30
    [-1, C2f, [256, 512]],  # 63
    [-1, C2f, [512, 512]],  # 64
    [-1, C2f, [512, 512]],  # 65


    #seg lane line
    [24, nn.Upsample, [None, 2, 'nearest']],  # 66  1/8
    [[-1, 19], Concat, [1]],  # 67  # cat backbone P4
    [-1, C2f, [1536, 512]],  # 68
    [-1, C2f, [512, 512]],  # 69
    [-1, C2f, [512, 512]],  # 70

    [-1, nn.Upsample, [None, 2, 'nearest']],  # 71  1/4
    [[-1, 12], Concat, [1]],  # 72 # cat backbone P3
    [-1, Conv, [768, 256, 3, 1]],  # 73 1/4 - 2

    [-1, C2f, [256, 256]],  # 74
    [-1, nn.Upsample, [None, 2, 'nearest']],  # 75 1/2 - 4
    [-1, Conv, [256, 256, 3, 1]],  # 76  1/2 - 6

    [-1, C2f, [256, 256]],  # 77
    [-1, Upsample, [None, 2, 'nearest']],  # 78  1 - 12
    [-1, Conv, [256, 256, 3, 1]],  # 79 1 - 14
    [-1, C2f, [256, 256]],  # 80

    [-1, Upsample, [None, 2, 'nearest']],  # 61  2 - 28
    [-1, Conv, [256, 256, 3, 1]],  # 62 2 - 30
    [-1, C2f, [256, 512]],  # 63
    [-1, C2f, [512, 512]],  # 64
    [-1, C2f, [512, 512]],  # 65
]

YOLODSs = [
    [25, 41, 57],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],    # 0 p1
    [-1, Conv, [32, 64, 3, 2]],     # 1  1/2
    [-1, Conv, [64, 128, 1, 1]],    # 2 p2
    [-1, C2f, [128, 128, True]],     # 3
    [-1, Conv, [128, 256, 3, 2]],    # 4 p3  1/4
    [-1, C2f, [256, 256, True]],     # 5
    [-1, C2f, [256, 256, True]],     # 6
    [-1, Conv, [256, 512, 3, 2]],    # 7 p4  1/8
    [-1, C2f, [512, 512, True]],     # 8
    [-1, C2f, [512, 512, True]],     # 9
    [-1, Conv, [512, 1024, 3, 2]],   # 10 p5  1/16
    [-1, C2f, [1024, 1024, True]],    # 11
    [-1, SPPF, [1024, 1024, 5]],      # 12

    [-1, nn.Upsample, [None, 2, 'nearest']],      #13  1/8
    [[-1, 9], Concat, [1]],        #14  # cat backbone P4
    [-1, C2f, [1536, 512]],       # 15

    [-1, nn.Upsample, [None, 2, 'nearest']],        # 16  1/4
    [[-1, 6], Concat, [1]],     # 17 # cat backbone P3
    [-1, C2f, [768, 256]],       # 18

    [-1, Conv, [256, 256, 3, 2]],        # 19  1/8
    [[-1, 9], Concat, [1]],        # 20
    [-1, C2f, [768, 512]],       # 21

    [-1, Conv, [512, 512, 3, 2]],        # 22  1/16
    [[-1, 12], Concat, [1]],        # 23  # cat head P5
    [-1, C2f, [1536, 1024]],      # 24

    [ [18, 21, 24], Detect,
      [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]],
       [256, 512, 1024]]], #Detection head 25

    # seg drive area
    [12, nn.Upsample, [None, 2, 'nearest']],  # 26  1/8
    [[-1, 9], Concat, [1]],  # 27  # cat backbone P4
    [-1, C2f, [1536, 512]],  # 28

    [-1, nn.Upsample, [None, 2, 'nearest']],  # 29  1/4
    [[-1, 6], Concat, [1]],  # 30 # cat backbone P3
    [-1, Conv, [768, 256, 3, 1]],   # 31 1/4 - 2

    [-1, C2f, [256, 256]],  # 32
    [-1, nn.Upsample, [None, 2, 'nearest']],    # 33 1/2 - 4
    [-1, Conv, [256, 256, 3, 1]],   # 34  1/2 - 6

    [-1, C2f, [256, 256]],  # 35
    [-1, Upsample, [None, 2, 'nearest']],  # 36  1 - 12
    [-1, Conv, [256, 256, 3, 1]],   # 37 1 - 14
    [-1, C2f, [256, 256]],  # 38

    [-1, Upsample, [None, 2, 'nearest']],  # 39  2 - 28
    [-1, Conv, [256, 256, 3, 1]],  # 40 2 - 30
    [-1, C2f, [256, 512]],  # 41



    #seg lane line
    [12, nn.Upsample, [None, 2, 'nearest']],  # 42  1/8
    [[-1, 9], Concat, [1]],  # 43  # cat backbone P4
    [-1, C2f, [1536, 512]],  # 44

    [-1, nn.Upsample, [None, 2, 'nearest']],  # 45  1/4
    [[-1, 6], Concat, [1]],  # 46 # cat backbone P3
    [-1, Conv, [768, 256, 3, 1]],   # 47 1/4 - 2

    [-1, C2f, [256, 256]],  # 48
    [-1, nn.Upsample, [None, 2, 'nearest']],    # 49 1/2 - 4
    [-1, Conv, [256, 256, 3, 1]],   # 50  1/2 - 6

    [-1, C2f, [256, 256]],  # 51
    [-1, Upsample, [None, 2, 'nearest']],  # 52  1 - 12
    [-1, Conv, [256, 256, 3, 1]],   # 53 1 - 14
    [-1, C2f, [256, 256]],  # 54

    [-1, Upsample, [None, 2, 'nearest']],  # 55  2 - 28
    [-1, Conv, [256, 256, 3, 1]],  # 56 2 - 30
    [-1, C2f, [256, 512]],  # 57
]

YOLODSn = [
    [18, 31, 44],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 16, 3]],    # 0 p1
    [-1, Conv, [16, 32, 3, 2]],     # 1  1/2
    [-1, Conv, [32, 64, 1, 1]],    # 2 p2
    [-1, C2f, [64, 64, True]],     # 3
    [-1, Conv, [64, 128, 3, 2]],    # 4 p3  1/4
    [-1, C2f, [128, 128, True]],     # 5
    [-1, Conv, [128, 256, 3, 2]],    # 6 p4  1/8
    [-1, C2f, [256, 256, True]],     # 7
    [-1, SPPF, [256, 256, 5]],      # 8

    [-1, nn.Upsample, [None, 2, 'nearest']],        # 9  1/4
    [[-1, 5], Concat, [1]],     # 10 # cat backbone P3
    [-1, C2f, [384, 128]],       # 11

    [-1, nn.Upsample, [None, 2, 'nearest']],  # 12 1/2
    [[-1, 3], Concat, [1]],        # 13 p2
    [-1, C2f, [192, 256]],       # 14

    [-1, nn.Upsample, [None, 2, 'nearest']],  # 15 1
    [[-1, 0], Concat, [1]],        # 16  # cat head P1
    [-1, C2f, [272, 256]],      # 17

    [ [11, 14, 17], Detect,
      [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]],
       [128, 256, 256]]], #Detection head 18

    # seg drive area
    [8, nn.Upsample, [None, 2, 'nearest']],  # 19  1/4
    [[-1, 5], Concat, [1]],  # 20 # cat backbone P3
    [-1, Conv, [384, 256, 3, 1]],   # 21 1/4 - 2

    [-1, C2f, [256, 128]],  # 22
    [-1, nn.Upsample, [None, 2, 'nearest']],    # 23 1/2 - 4
    [-1, Conv, [128, 64, 3, 1]],   # 24  1/2 - 6

    [-1, C2f, [64, 32]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26  1 - 12
    [-1, Conv, [32, 16, 3, 1]],   # 27 1 - 14
    [-1, C2f, [16, 8]],  # 28

    [-1, Upsample, [None, 2, 'nearest']],  # 29  2 - 28
    [-1, Conv, [8, 4, 3, 1]],  # 30 2 - 30
    [-1, C2f, [4, 2]],  # 31


    #seg lane line
    [8, nn.Upsample, [None, 2, 'nearest']],  # 19  1/4
    [[-1, 5], Concat, [1]],  # 20 # cat backbone P3
    [-1, Conv, [384, 256, 3, 1]],   # 21 1/4 - 2

    [-1, C2f, [256, 128]],  # 22
    [-1, nn.Upsample, [None, 2, 'nearest']],    # 23 1/2 - 4
    [-1, Conv, [128, 64, 3, 1]],   # 24  1/2 - 6

    [-1, C2f, [64, 32]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26  1 - 12
    [-1, Conv, [32, 16, 3, 1]],   # 27 1 - 14
    [-1, C2f, [16, 8]],  # 28

    [-1, Upsample, [None, 2, 'nearest']],  # 29  2 - 28
    [-1, Conv, [8, 4, 3, 1]],  # 30 2 - 30
    [-1, C2f, [4, 2]],  # 31
]

class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]


        # Build model
        # print(list(enumerate(block_cfg[1:])))
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out


    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs):
    # m_block_cfg = YOLOP
    m_block_cfg = YOLODSn
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    # from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out,SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)

