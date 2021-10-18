#!/usr/bin/env python
from collections import namedtuple
from typing import Generic, TypeVar

import numpy as np

__all__ = ["Array", "BBoxXYXY", "BBoxXYWH"]

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    pass


BBoxXYXY = namedtuple("BBoxXYXY", "x_min y_min x_max y_max")

BBoxXYWH = namedtuple("BBoxXYWH", "x_min y_min width height")
