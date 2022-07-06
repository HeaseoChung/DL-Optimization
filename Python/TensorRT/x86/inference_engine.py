import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch


class HostDeviceMem(object):
    """CPU & CPU의 메모리를 설정하고 주소를 반환하는 클래스"""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        """CPU (host) & GPU (host)의 주소를 문자열로 반환하는 메소드"""
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        """CPU (host) & GPU (host)의 주소를 반환하는 메소드"""
        return self.__str__()


class EngineInferencer(object):
    def __init__(self, trt_engine_path):
        """trt engine 셋팅"""
        # trt 로그 초기화
        TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
        # runtime에 로그 추가
        trt_runtime = trt.Runtime(TRT_LOGGER)
        # CPU & GPU의 원활한 통신을 하기 위해 stream 사용
        self.stream = cuda.Stream()
        # .trt 파일 로드 후 trt_runtime에 추가
        self.engine = self.load_engine(trt_runtime, trt_engine_path)
        # Inference 할 context 준비
        self.context = self.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []

    def load_engine(self, trt_runtime, engine_path):
        """엔진 로드 메소드"""
        # Serialized 된 엔진 데이터 읽기모드
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        # Serialized 된 엔진 데이터를 다시 객체형태로 복원하여 엔진으로 생성
        return trt_runtime.deserialize_cuda_engine(engine_data)

    def create_execution_context(self):
        """Inference 할 context를 만드는 메소드"""
        return self.engine.create_execution_context()

    def do_inference(self, frame):
        """추론 메소드"""
        # 호스트 설정
        hosts = [input.host for input in self.inputs]

        # 호스트에 lr 이미지 복사
        for host in hosts:
            # lr 이미지를 float16 데이터타입을 가진 1차원으로 변형
            numpy_array = (
                np.asarray(frame).astype(trt.nptype(trt.float32)).ravel()
            )
            np.copyto(host, numpy_array)

        # Transfer input data to the GPU.
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            for inp in self.inputs
        ]
        # Run inference.
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            for out in self.outputs
        ]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return self.outputs[0].host

    def dynamicBindingProcess(self, height, width, scale, channel=3):
        for i, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(i, (1, channel, height, width))
            else:
                self.context.set_binding_shape(
                    i, (1, channel, height * scale, width * scale)
                )

            size = trt.volume(self.context.get_binding_shape(i)) * 1
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # CPU 메모리 사이즈 설정
            host_mem = cuda.pagelocked_empty(size, dtype)
            # GPU 메모리 사이즈 설정
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def staticsbindingProcess(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * 1
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # CPU 메모리 사이즈 설정
            host_mem = cuda.pagelocked_empty(size, dtype)
            # GPU 메모리 사이즈 설정
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
        return self.inputs, self.outputs, self.bindings
