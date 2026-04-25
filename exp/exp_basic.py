import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use multi GPU devices: {}, active cuda:{}'.format(self.args.devices, self.args.gpu))
            else:
                # Respect externally forced CUDA_VISIBLE_DEVICES (e.g. export CUDA_VISIBLE_DEVICES=3).
                visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                if not visible_devices:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]

                # Under single-GPU visibility, the valid logical index is cuda:0.
                device = torch.device('cuda:0')
                print('Use single GPU: cuda:0 (CUDA_VISIBLE_DEVICES={})'.format(visible_devices))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
