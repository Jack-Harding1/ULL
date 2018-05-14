#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:12:48 2018

@author: jackharding
"""

import torch.nn.functional as F

test1 = torch.tensor([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [0, 0, 0],
                     [0, 1, 0]], dtype=torch.float)

print(test1.size())
m = F.log_softmax(test1, dim=0)
print(m)