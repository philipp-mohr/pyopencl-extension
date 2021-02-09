from pyopencl_extension.modifications_pyopencl import cltypes
from dataclasses import dataclass
import numpy as np


@dataclass
class ClTypesVector:
    char2: np.dtype = cltypes.char2
    char4: np.dtype = cltypes.char4
    char8: np.dtype = cltypes.char8
    char16: np.dtype = cltypes.char16
    short2: np.dtype = cltypes.short2
    short4: np.dtype = cltypes.short4
    short8: np.dtype = cltypes.short8
    short16: np.dtype = cltypes.short16
    int2: np.dtype = cltypes.int2
    int4: np.dtype = cltypes.int4
    int8: np.dtype = cltypes.int8
    int16: np.dtype = cltypes.int16
    long2: np.dtype = cltypes.long2
    long4: np.dtype = cltypes.long4
    long8: np.dtype = cltypes.long8
    long16: np.dtype = cltypes.long16
    uchar2: np.dtype = cltypes.uchar2
    uchar4: np.dtype = cltypes.uchar4
    uchar8: np.dtype = cltypes.uchar8
    uchar16: np.dtype = cltypes.uchar16
    ushort2: np.dtype = cltypes.ushort2
    ushort4: np.dtype = cltypes.ushort4
    ushort8: np.dtype = cltypes.ushort8
    ushort16: np.dtype = cltypes.ushort16
    uint2: np.dtype = cltypes.uint2
    uint4: np.dtype = cltypes.uint4
    uint8: np.dtype = cltypes.uint8
    uint16: np.dtype = cltypes.uint16
    ulong2: np.dtype = cltypes.ulong2
    ulong4: np.dtype = cltypes.ulong4
    ulong8: np.dtype = cltypes.ulong8
    ulong16: np.dtype = cltypes.ulong16
    half2: np.dtype = cltypes.half2
    half4: np.dtype = cltypes.half4
    half8: np.dtype = cltypes.half8
    half16: np.dtype = cltypes.half16
    float2: np.dtype = cltypes.float2
    float4: np.dtype = cltypes.float4
    float8: np.dtype = cltypes.float8
    float16: np.dtype = cltypes.float16
    double2: np.dtype = cltypes.double2
    double4: np.dtype = cltypes.double4
    double8: np.dtype = cltypes.double8
    double16: np.dtype = cltypes.double16


@dataclass
class ClTypesScalar:
    char: np.dtype = cltypes.char
    short: np.dtype = cltypes.short
    int: np.dtype = cltypes.int
    long: np.dtype = cltypes.long
    uchar: np.dtype = cltypes.uchar
    ushort: np.dtype = cltypes.ushort
    uint: np.dtype = cltypes.uint
    ulong: np.dtype = cltypes.ulong
    half: np.dtype = cltypes.half
    float: np.dtype = cltypes.float
    double: np.dtype = cltypes.double


@dataclass
class _ClTypes(ClTypesScalar, ClTypesVector):
    pass
