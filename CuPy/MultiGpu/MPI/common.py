import time

import numpy as np
import cupy as cp
from cupy.cuda import Device
from cupy.cuda.runtime import getDeviceCount

class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        self.end_time = time.perf_counter()

    def get_time(self):
        return self.end_time - self.start_time
    
class MultiGpu(object):
    # numbers
	begin: int = 0
	end: int = 0
	num_of_gpu: int = 0
	# dimentional size
	N: int = 0
	local_N: int = 0
	# matrix
	A: list = []
	# vector
	x: list = []
	y: list = []
	out: list = []
	# streams
	streams = None

	# GPUの初期化
	@classmethod
	def init(cls):
		cls.begin = 0
		cls.end = getDeviceCount()
		cls.num_of_gpu = getDeviceCount()
		cls.streams = [None] * cls.num_of_gpu

		# init memory allocator
		for i in range(cls.num_of_gpu):
			Device(i).use()
			pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
			cp.cuda.set_allocator(pool.malloc)
			cls.streams[i] = cp.cuda.Stream()

			# Enable P2P
			for j in range(cls.num_of_gpu):
				if i == j:
					continue
				cp.cuda.runtime.deviceEnablePeerAccess(j)
	
	# allocate A, x, y
	@classmethod
	def alloc(cls, A, b, T):
		# dimentional size
		cls.N = b.size
		cls.local_N = cls.N // cls.num_of_gpu
		# bite size
		cls.nbytes = b.nbytes
		cls.local_nbytes = b.nbytes // cls.num_of_gpu

		# init list
		cls.A = [None] * cls.num_of_gpu
		cls.x = [None] * cls.num_of_gpu
		cls.y = [None] * cls.num_of_gpu

		# allocate A, x, y
		for i in range(cls.num_of_gpu):
			Device(i).use()
			# device A
			if isinstance(A, np.ndarray):
				cls.A[i] = cp.array(A[i*cls.local_N:(i+1)*cls.local_N], T)
			else:
				from cupyx.scipy.sparse import csr_matrix
				cls.A[i] = csr_matrix(A[i*cls.local_N:(i+1)*cls.local_N])
			cls.x[i] = cp.zeros(cls.N, T)
			cls.y[i] = cp.zeros(cls.local_N, T)

		# allocate output vector
		cls.out = cp.zeros(cls.N, T)
  
		# matvec with multi-gpu
		@classmethod
		def dot(cls, A, x):
			# copy to workers
			for i in range(cls.num_of_gpu):
				Device(i).use()
				cp.cuda.runtime.memcpyPeerAsync(cls.x[i].data.ptr, i, x.data.ptr, cls.end, cls.nbytes, cls.streams[i].ptr)
				# dot
				cls.y[i] = cls.A[i].dot(cls.x[i])
			# copy to master
			for i in range(cls.num_of_gpu):
				cp.cuda.runtime.memcpyPeerAsync(cls.out[i*cls.local_N].data.ptr, cls.end, cls.y[i].data.ptr, i, cls.local_nbytes, cls.streams[i].ptr)

			# sync
			for i in range(cls.num_of_gpu):
				cls.streams[i].synchronize()
			return cls.out