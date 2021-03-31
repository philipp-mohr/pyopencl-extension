from pyopencl_extension.modifications_pyopencl import cltypes
from dataclasses import dataclass
import numpy as np
from typing import Callable, Union

@dataclass(frozen=True)
class ClTypesVector:
   char2:Union[np.dtype, Callable]=cltypes.char2
   char4:Union[np.dtype, Callable]=cltypes.char4
   char8:Union[np.dtype, Callable]=cltypes.char8
   char16:Union[np.dtype, Callable]=cltypes.char16
   short2:Union[np.dtype, Callable]=cltypes.short2
   short4:Union[np.dtype, Callable]=cltypes.short4
   short8:Union[np.dtype, Callable]=cltypes.short8
   short16:Union[np.dtype, Callable]=cltypes.short16
   int2:Union[np.dtype, Callable]=cltypes.int2
   int4:Union[np.dtype, Callable]=cltypes.int4
   int8:Union[np.dtype, Callable]=cltypes.int8
   int16:Union[np.dtype, Callable]=cltypes.int16
   long2:Union[np.dtype, Callable]=cltypes.long2
   long4:Union[np.dtype, Callable]=cltypes.long4
   long8:Union[np.dtype, Callable]=cltypes.long8
   long16:Union[np.dtype, Callable]=cltypes.long16
   uchar2:Union[np.dtype, Callable]=cltypes.uchar2
   uchar4:Union[np.dtype, Callable]=cltypes.uchar4
   uchar8:Union[np.dtype, Callable]=cltypes.uchar8
   uchar16:Union[np.dtype, Callable]=cltypes.uchar16
   ushort2:Union[np.dtype, Callable]=cltypes.ushort2
   ushort4:Union[np.dtype, Callable]=cltypes.ushort4
   ushort8:Union[np.dtype, Callable]=cltypes.ushort8
   ushort16:Union[np.dtype, Callable]=cltypes.ushort16
   uint2:Union[np.dtype, Callable]=cltypes.uint2
   uint4:Union[np.dtype, Callable]=cltypes.uint4
   uint8:Union[np.dtype, Callable]=cltypes.uint8
   uint16:Union[np.dtype, Callable]=cltypes.uint16
   ulong2:Union[np.dtype, Callable]=cltypes.ulong2
   ulong4:Union[np.dtype, Callable]=cltypes.ulong4
   ulong8:Union[np.dtype, Callable]=cltypes.ulong8
   ulong16:Union[np.dtype, Callable]=cltypes.ulong16
   half2:Union[np.dtype, Callable]=cltypes.half2
   half4:Union[np.dtype, Callable]=cltypes.half4
   half8:Union[np.dtype, Callable]=cltypes.half8
   half16:Union[np.dtype, Callable]=cltypes.half16
   float2:Union[np.dtype, Callable]=cltypes.float2
   float4:Union[np.dtype, Callable]=cltypes.float4
   float8:Union[np.dtype, Callable]=cltypes.float8
   float16:Union[np.dtype, Callable]=cltypes.float16
   double2:Union[np.dtype, Callable]=cltypes.double2
   double4:Union[np.dtype, Callable]=cltypes.double4
   double8:Union[np.dtype, Callable]=cltypes.double8
   double16:Union[np.dtype, Callable]=cltypes.double16

@dataclass(frozen=True)
class ClTypesScalar:
   char:Union[np.dtype, Callable]=cltypes.char
   short:Union[np.dtype, Callable]=cltypes.short
   int:Union[np.dtype, Callable]=cltypes.int
   long:Union[np.dtype, Callable]=cltypes.long
   uchar:Union[np.dtype, Callable]=cltypes.uchar
   ushort:Union[np.dtype, Callable]=cltypes.ushort
   uint:Union[np.dtype, Callable]=cltypes.uint
   ulong:Union[np.dtype, Callable]=cltypes.ulong
   half:Union[np.dtype, Callable]=cltypes.half
   float:Union[np.dtype, Callable]=cltypes.float
   double:Union[np.dtype, Callable]=cltypes.double

@dataclass(frozen=True)
class _ClTypes(ClTypesScalar, ClTypesVector):
   pass
