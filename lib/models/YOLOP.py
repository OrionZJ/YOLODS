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
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, C2f, SPPF, \
    SimCSPSPPF, SPPFCSPC, C3k2, C2PSA
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized


YOLODSn = [
    [19, 32, 45],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],    # 0 p1  2 -> 1 640->320
    [-1, Conv, [32, 64, 3, 2]],     # 1  1/2
    [-1, Conv, [64, 128, 1, 1]],    # 2 p2
    [-1, C3k2, [128, 128, False]],     # 3
    [-1, Conv, [128, 256, 3, 2]],    # 4 p3  1/4
    [-1, C3k2, [256, 256, True]],     # 5
    [-1, Conv, [256, 512, 3, 2]],    # 6 p4  1/8
    [-1, C3k2, [512, 512, True]],     # 7
    [-1, SPPF, [512, 512, 5]],      # 8
    [-1, C2PSA, [512, 512]],
    # [-1, SimCSPSPPF, [512, 512, 5]],      # 8 SimCSPSPPF

    [-1, nn.Upsample, [None, 2, 'nearest']],        # 9  1/4
    [[-1, 5], Concat, [1]],     # 10 # cat backbone P3
    [-1, C2f, [768, 128]],       # 11

    [-1, nn.Upsample, [None, 2, 'nearest']],  # 12 1/2
    [[-1, 3], Concat, [1]],        # 13 cat backbone p2
    [-1, C2f, [256, 256]],       # 14

    # [-1, nn.Upsample, [None, 2, 'nearest']],  # 15 1
    [-1, Conv, [256, 512, 3, 2]],   # 15 1/4
    [[-1, 12], Concat, [1]],        # 16  # cat head 11
    [-1, C2f, [640, 512]],      # 17

    [ [12, 15, 18], Detect,
      [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]],
       [128, 256, 512]]], #Detection head 18

    # seg drive area
    [8, nn.Upsample, [None, 2, 'nearest']],  # 19  1/4
    [[-1, 5], Concat, [1]],  # 20 # cat backbone P3
    [-1, Conv, [768, 256, 3, 1]],   # 21 1/4 - 2

    [-1, C3k2, [256, 128, False]],  # 22
    [-1, nn.Upsample, [None, 2, 'nearest']],    # 23 1/2 - 4
    [-1, Conv, [128, 64, 3, 1]],   # 24  1/2 - 6

    [-1, C3k2, [64, 32, False]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26  1 - 12
    [-1, Conv, [32, 16, 3, 1]],   # 27 1 - 14
    [-1, C3k2, [16, 8, True]],  # 28
 
    [-1, Upsample, [None, 2, 'nearest']],  # 29  2 - 28
    [-1, Conv, [8, 4, 3, 1]],  # 30 2 - 30
    [-1, C3k2, [4, 2, False]],  # 31


    #seg lane line
    [8, nn.Upsample, [None, 2, 'nearest']],  # 19  1/4
    [[-1, 5], Concat, [1]],  # 20 # cat backbone P3
    [-1, Conv, [768, 256, 3, 1]],   # 21 1/4 - 2

    [-1, C3k2, [256, 128, False]],  # 22
    [-1, nn.Upsample, [None, 2, 'nearest']],    # 23 1/2 - 4
    [-1, Conv, [128, 64, 3, 1]],   # 24  1/2 - 6

    [-1, C3k2, [64, 32, False]],  # 25
    [-1, Upsample, [None, 2, 'nearest']],  # 26  1 - 12
    [-1, Conv, [32, 16, 3, 1]],   # 27 1 - 14
    [-1, C3k2, [16, 8, True]],  # 28

    [-1, Upsample, [None, 2, 'nearest']],  # 29  2 - 28
    [-1, Conv, [8, 4, 3, 1]],  # 30 2 - 30
    [-1, C3k2, [4, 2, False]],  # 31
]

YOLOPs = [
    [22, 35, 48],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx
    # [-1, Focus, [3, 32, 3]],    #
    [-1, Conv, [3, 32, 3, 2]],  # 0  1/2  640->320
    [-1, Conv, [32, 64, 3, 2]],  # 1  1/4 p2
    [-1, C2f, [64, 64, True]],  # 2
    [-1, Conv, [64, 128, 3, 2]],  # 3 p3  1/8
    [-1, C2f, [128, 128, True]],  # 4
    [-1, Conv, [128, 256, 3, 2]],  # 5 p4  1/16
    [-1, C2f, [256, 256, True]],  # 6
    [-1, Conv, [256, 512, 3, 2]],  # 7 p5  1/32
    [-1, C2f, [512, 512, True]],  # 8
    # [-1, SPPF, [512, 512, 5]],
    [-1, SPPFCSPC, [512, 512, 5]],
    # [-1, SimCSPSPPF, [512, 512, 5]],  # 9 SimCSPSPPF

    [-1, Upsample, [None, 2, 'nearest']],  # 10  1/16
    [[-1, 6], Concat, [1]],  # 11 # cat backbone P3
    [-1, C2f, [768, 256]],  # 12

    [-1, Upsample, [None, 2, 'nearest']],  # 13 1/8
    [[-1, 4], Concat, [1]],  # 14 cat backbone p2
    [-1, C2f, [384, 128]],  # 15

    [-1, Conv, [128, 128, 3, 2]],  # 16 1/16
    [[-1, 12], Concat, [1]],  # 17  # cat p4
    [-1, C2f, [384, 256]],  # 18

    [-1, Conv, [256, 256, 3, 2]],  # 19 1/32
    [[-1, 9], Concat, [1]],  # 20  cat sppf
    [-1, C2f, [768, 512]],  # 21

    [[15, 18, 21], Detect,
     [1, [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]],
      [128, 256, 512]]],  # Detection head 22

    # seg drive area
    [9, Upsample, [None, 4, 'nearest']],  # 23  1/8
    [[-1, 4], Concat, [1]],  # 24 # cat backbone P3
    [-1, Conv, [640, 256, 3, 1]],  # 25 1/8 - 2

    [-1, C2f, [256, 128]],  # 26
    [-1, Upsample, [None, 2, 'nearest']],  # 27 1/4 - 4
    [-1, Conv, [128, 64, 3, 1]],  # 28  1/4 - 6

    [-1, C2f, [64, 32]],  # 29
    [-1, Upsample, [None, 2, 'nearest']],  # 30  1/2 - 12
    [-1, Conv, [32, 16, 3, 1]],  # 31 1/2 - 14
    [-1, C2f, [16, 8]],  # 32

    [-1, Upsample, [None, 2, 'nearest']],  # 33  1 - 28
    [-1, Conv, [8, 4, 3, 1]],  # 34 1 - 30
    [-1, C2f, [4, 2]],  # 35

    # seg lane line
    [9, nn.Upsample, [None, 4, 'nearest']],  # 36  1/8
    [[-1, 4], Concat, [1]],  # 37 # cat backbone P3
    [-1, Conv, [640, 256, 3, 1]],  # 38 1/8 - 2

    [-1, C2f, [256, 128]],  # 39
    [-1, nn.Upsample, [None, 2, 'nearest']],  # 40 1/4 - 4
    [-1, Conv, [128, 64, 3, 1]],  # 41  1/4 - 6

    [-1, C2f, [64, 32]],  # 42
    [-1, Upsample, [None, 2, 'nearest']],  # 43  1/2 - 12
    [-1, Conv, [32, 16, 3, 1]],  # 44 1/2 - 14
    [-1, C2f, [16, 8]],  # 45

    [-1, Upsample, [None, 2, 'nearest']],  # 46  1 - 28
    [-1, Conv, [8, 4, 3, 1]],  # 47 1 - 30
    [-1, C2f, [4, 2]],  # 48
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

