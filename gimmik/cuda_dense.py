# -*- coding: utf-8 -*-

from gimmik.base import MatMul
import numpy as np


class CUDADenseMatMul(MatMul):
    platform = 'cuda'
    basemeta = {'block': (128, 1, 1), 'width': 1, 'shared': 0,
                'dynamic_shared': 0}

    def _kernel_generators(self, dtype, dsize, *, compute_capability=None):
        # B streaming, C accumulation kernel
        self.afix_1 = (self.mat_1 != 0).argmax(axis=1)
        self.afix_1 = np.where(np.any(self.mat_1 != 0, axis=1), self.afix_1, -1)
        self.afix_2 = (self.mat_2 != 0).argmax(axis=1)
        self.afix_2 = np.where(np.any(self.mat_2 != 0, axis=1), self.afix_2, -1)
        args = {'A_1': self.mat_1, 'A_2': self.mat_2, 'afix_1': self.afix_1, 'afix_2': self.afix_2}
        yield ('bstream-3-mat', args, {})

    def _process_meta(self, meta):
        if self.n is not None:
            div = meta['block'][0]*meta['width']
            meta['grid'] = (-(-self.n // div), 1, 1)
