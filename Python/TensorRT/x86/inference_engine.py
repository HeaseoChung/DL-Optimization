import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem:
    """CPU & CPU의 메모리를 설정하고 주소를 반환하는 클래스"""

    def __init__(
        self, host_mem: np.ndarray, device_mem: pycuda._driver.DeviceAllocation
    ):
        """
        Args:
            host_mem(np.ndarry): input 데이터
            device_mem(pycuda._driver.DeviceAllocation): linear 디바이스 메모리
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        """CPU (host) & GPU (host)의 주소를 문자열로 반환하는 메서드
        Returns:
            str: cpu 및 gpu에 할당된 데이터를 str으로 표기하여 반환
        """
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        """CPU (host) & GPU (host)의 주소를 반환하는 메소드
        Returns:
            str: cpu 및 gpu에 할당된 데이터를 str으로 표기하여 반환
        """
        return self.__str__()


class EngineInferencer:
    """Trt 모델을 추론하기 위해 필요한 기능들을 포함하고 있는 클래스"""

    def __init__(self, trt_engine_path: str):
        """EngineInferencer 클래스의 init 메서드
        Args:
            trt_engine_path(str): TensorRT 모델의 경로
        """
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

    def load_engine(
        self, trt_runtime: trt.tensorrt.Runtime, engine_path: str
    ) -> trt.tensorrt.ICudaEngine:
        """엔진 로드 메서드
        Args:
            trt_runtime(object): trt runtime 클래스
            engine_path(str): TensorRT 모델의 경로
        Returns:
            obejct(tensorrt.tensorrt.ICudaEngine): Serialized 된 trt 모델 데이터를 객체로 변환하여 반환
        """

        # Serialized 된 엔진 데이터 읽기모드
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        # Serialized 된 엔진 데이터를 다시 객체형태로 복원하여 엔진으로 생성

        return trt_runtime.deserialize_cuda_engine(engine_data)

    def create_execution_context(self) -> trt.tensorrt.IExecutionContext:
        """Inference 할 context를 만드는 메서드
        Returns:
            trt.tensorrt.IExecutionContext: 추론에 필요한 context 생성하고 반환
        """
        return self.engine.create_execution_context()

    def do_inference(self, frame: np.ndarray) -> np.ndarray:
        """추론 메서드
        Args:
            frame (numpy.ndarray): input 이미지 데이터
        Returns:
            self.outputs[0].host(numpy.ndarray): 추론 결과 데이터를 반환
        """

        # 호스트 설정
        hosts = [input.host for input in self.inputs]

        # 호스트에 lr 이미지 복사
        for host in hosts:
            # lr 이미지를 float16 데이터타입을 가진 1차원으로 변형
            numpy_array = (
                np.asarray(frame).astype(trt.nptype(trt.float32)).ravel()
            )
            np.copyto(host, numpy_array)

        # CPU에서 GPU로 데이터 전송
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            for inp in self.inputs
        ]
        # 추론
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        # GPU에서 CPU로 데이터 전송
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            for out in self.outputs
        ]

        self.stream.synchronize()
        return self.outputs[0].host

    def setMemBindingProcess(self, size: int, dtype: type, binding: str):
        """cpu & gpu 에 할당 된 Input & Output 데이터의 주소를 지정하는 메서드
        Args:
            size(int): 데이터의 사이즈 크기
            dtype(type): 데이터 타입
            binding(str): 데이터의 이름
        """

        # CPU 메모리 사이즈 설정
        host_mem = cuda.pagelocked_empty(size, dtype)
        # GPU 메모리 사이즈 설정
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        self.bindings.append(int(device_mem))

        if self.engine.binding_is_input(binding):
            # input의 cpu & gpu에 할당 된 데이터 및 주소 값 지정
            self.inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # output의 cpu & gpu에 할당 된 데이터 및 주소 값 지정
            self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def dynamicBindingProcess(
        self, height: int, width: int, scale: int, channel=3
    ):
        """동적으로 input 및 output 버퍼를 지정하는 메서드
        Args:
            height(int): 이미지의 세로
            width(int): 이미지의 가로
            scale(int): upsample의 scale 값
            channel(int): 이미지의 채널
        """

        for i, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                # trt context 에 input 의 shape 지정
                self.context.set_binding_shape(i, (1, channel, height, width))
            else:
                # trt context 에 output 의 shape 지정
                self.context.set_binding_shape(
                    i, (1, channel, height * scale, width * scale)
                )
            # 데이터의 size 및 type 지정
            size = trt.volume(self.context.get_binding_shape(i)) * 1
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # input 및 output 을 CPU 또는 GPU 메모리에 할당할 수 있는 주소값 지정
            self.setMemBindingProcess(size, dtype, binding)

    def staticsBindingProcess(self):
        """정적으로 input 및 output 버퍼를 지정하는 메서드"""
        for binding in self.engine:
            # 데이터의 size 및 type 지정
            size = trt.volume(self.engine.get_binding_shape(binding)) * 1
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # input 및 output 을 CPU 또는 GPU 메모리에 할당할 수 있는 주소값 지정
            self.setMemBindingProcess(size, dtype, binding)
