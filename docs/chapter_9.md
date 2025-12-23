---
sidebar_position: 9
title: "Chapter 9: Edge AI and Computing Platforms"
---

# Chapter 9: Edge AI and Computing Platforms

## Learning Outcomes
By the end of this chapter, students will be able to:
- Compare different edge AI computing platforms for robotics
- Optimize AI models for deployment on resource-constrained devices
- Implement real-time inference pipelines for robotic applications
- Evaluate computing platform performance for specific robotics tasks
- Design distributed computing architectures for humanoid robots
- Assess power consumption and thermal management requirements

## Overview

Edge AI computing platforms have revolutionized robotics by enabling on-board artificial intelligence capabilities without relying on cloud connectivity. For humanoid robots, which require real-time processing of sensor data and immediate response to environmental changes, edge AI platforms are essential. These platforms provide the computational power needed for perception, decision-making, and control while maintaining low latency and privacy.

The selection of an appropriate computing platform is critical for humanoid robotics applications, as it affects performance, power consumption, cost, and deployment flexibility. This chapter explores various edge AI platforms, their characteristics, and their applications in humanoid robotics.

## Edge AI Platform Characteristics

### Performance Metrics
Edge AI platforms for robotics are evaluated based on several key metrics:

#### Computational Performance
- **TOPS (Tera Operations Per Second)**: Measure of AI computational capability
- **FPS (Frames Per Second)**: Performance for vision tasks
- **Latency**: Time from input to output
- **Throughput**: Number of operations per unit time

#### Power Efficiency
- **TOPS/Watt**: Performance per unit of power consumption
- **Thermal Design Power (TDP)**: Maximum heat dissipation
- **Battery Life**: Operational time on battery power
- **Power Management**: Dynamic scaling and sleep modes

#### Hardware Specifications
- **CPU Cores**: General-purpose processing capability
- **GPU Cores**: Parallel processing for graphics and AI
- **NPU (Neural Processing Unit)**: Dedicated AI acceleration
- **Memory**: RAM and storage capacity
- **Connectivity**: WiFi, Bluetooth, Ethernet, CAN bus

### Platform Categories

#### High-Performance Platforms
- **NVIDIA Jetson Series**: GPU-accelerated AI computing
- **Intel Movidius**: Vision processing units
- **Google Coral**: Edge TPU for machine learning

#### Balanced Platforms
- **Raspberry Pi 4**: General-purpose with AI acceleration options
- **Jetson Nano**: Lower-cost NVIDIA platform
- **Odroid Series**: ARM-based high-performance computing

#### Ultra-Low Power Platforms
- **ESP32-S3**: Microcontroller with AI capabilities
- **STM32MP1**: ARM Cortex-A with AI extensions
- **Ambiq Apollo Series**: Ultra-low power AI processing

## NVIDIA Jetson Platform

### Jetson Family Overview
The NVIDIA Jetson family provides GPU-accelerated computing for AI applications:

#### Jetson Orin Series
- **Jetson Orin AGX**: 275 TOPS AI performance
- **Jetson Orin NX**: 100 TOPS AI performance
- **Jetson Orin Nano**: 40 TOPS AI performance

#### Jetson Xavier Series
- **Jetson AGX Xavier**: 32 TOPS AI performance
- **Jetson Xavier NX**: 21 TOPS AI performance

#### Jetson Nano
- **Performance**: 0.5 TOPS AI performance
- **Power**: 10W TDP
- **Cost-effective**: Entry-level AI acceleration

### Jetson Software Stack
```python
# Example: Jetson platform initialization
import jetson.inference
import jetson.utils
import cv2
import numpy as np
import time

class JetsonAIProcessor:
    def __init__(self, model_path=None, input_size=(224, 224)):
        self.input_size = input_size
        self.model_path = model_path
        self.device = 'cuda'

        # Initialize Jetson-specific optimizations
        self.initialize_jetson()

    def initialize_jetson(self):
        """Initialize Jetson-specific optimizations"""
        print(f"Jetson Platform Info:")
        print(f"- GPU: Available (CUDA)")
        print(f"- Tensor Cores: Available")
        print(f"- Memory: {self.get_gpu_memory()} MB")

    def get_gpu_memory(self):
        """Get available GPU memory"""
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total',
                                   '--format=csv,nounits,noheader'],
                                  capture_output=True, text=True)
            memory = result.stdout.strip().split('\n')[0]
            return int(memory)
        except:
            return 0

    def load_model(self, model_path):
        """Load model optimized for Jetson"""
        # Use TensorRT for optimization
        import tensorrt as trt
        # Implementation would use TensorRT optimization
        pass

    def run_inference(self, image):
        """Run inference on image"""
        # Convert image to CUDA memory
        cuda_img = jetson.utils.cudaFromNumpy(image)

        # Run inference
        start_time = time.time()
        result = self.model.Classify(cuda_img, self.input_size)
        end_time = time.time()

        inference_time = (end_time - start_time) * 1000  # ms

        return result, inference_time
```

### Jetson for Humanoid Robotics
```python
# Example: Humanoid perception system on Jetson
import jetson.inference
import jetson.utils
import cv2
import numpy as np
import threading
import queue

class JetsonHumanoidPerception:
    def __init__(self):
        # Initialize detection models
        self.detection_model = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
        self.segmentation_model = jetson.inference.segNet("fcn-resnet18-dinnerware")

        # Initialize pose estimation
        self.pose_model = jetson.inference.poseNet("resnet18-body", threshold=0.15)

        # Data queues
        self.image_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)

        # Performance tracking
        self.fps = 0
        self.inference_times = []

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.running = True
        self.processing_thread.start()

    def process_loop(self):
        """Continuous processing loop"""
        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                # Get image from queue
                if not self.image_queue.empty():
                    image = self.image_queue.get_nowait()

                    # Convert to CUDA memory
                    cuda_img = jetson.utils.cudaFromNumpy(image)

                    # Run object detection
                    detections = self.detection_model.Detect(cuda_img, image.shape[1], image.shape[0])

                    # Run pose estimation
                    poses = self.pose_model.Process(cuda_img, image.shape[1], image.shape[0])

                    # Calculate FPS
                    frame_count += 1
                    current_time = time.time()
                    if current_time - start_time >= 1.0:
                        self.fps = frame_count
                        frame_count = 0
                        start_time = current_time

                    # Store results
                    result = {
                        'detections': detections,
                        'poses': poses,
                        'fps': self.fps
                    }

                    if not self.result_queue.full():
                        self.result_queue.put(result)

            except queue.Empty:
                time.sleep(0.001)  # Small delay to prevent busy waiting

    def process_image(self, image):
        """Process image for humanoid perception"""
        if not self.image_queue.full():
            self.image_queue.put(image.copy())

        # Get latest results
        try:
            if not self.result_queue.empty():
                return self.result_queue.get_nowait()
        except queue.Empty:
            pass

        return None

    def stop(self):
        """Stop processing"""
        self.running = False
        self.processing_thread.join()
```

## Intel Movidius and Vision Processing Units

### Movidius Neural Compute Stick
- **Performance**: 4 TOPS
- **Power**: 1W
- **Connectivity**: USB 3.0
- **Use Case**: Vision processing, object detection

### OpenVINO Toolkit
```python
# Example: OpenVINO implementation for vision processing
from openvino.runtime import Core
import cv2
import numpy as np

class OpenVINOProcessor:
    def __init__(self, model_path):
        self.core = Core()
        self.model = self.core.read_model(model=model_path)
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU")

        # Get input and output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        # Model input shape
        self.input_shape = self.input_layer.shape
        self.n, self.c, self.h, self.w = self.input_shape

    def preprocess_image(self, image):
        """Preprocess image for OpenVINO model"""
        # Resize image
        resized_image = cv2.resize(image, (self.w, self.h))

        # Change data layout from HWC to CHW
        input_image = resized_image.transpose(2, 0, 1)

        # Expand dims to add batch dimension
        input_image = np.expand_dims(input_image, axis=0)

        return input_image

    def run_inference(self, image):
        """Run inference using OpenVINO"""
        preprocessed_image = self.preprocess_image(image)

        # Run inference
        result = self.compiled_model([preprocessed_image])[self.output_layer]

        return result

    def process_detection(self, image):
        """Process object detection"""
        # Preprocess image
        input_data = self.preprocess_image(image)

        # Run inference
        result = self.compiled_model([input_data])[self.output_layer]

        # Process results (implementation depends on model)
        # Return detection results
        return result
```

## Google Coral Edge TPU

### Edge TPU Characteristics
- **Performance**: 4 TOPS
- **Power**: 0.5W
- **Architecture**: Custom ASIC for TensorFlow Lite
- **Use Case**: Machine learning inference

### Coral Development
```python
# Example: Google Coral implementation
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

class CoralProcessor:
    def __init__(self, model_path):
        # Initialize Coral TPU
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate('libdelegate.so')]
        )
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image):
        """Preprocess image for Coral TPU"""
        # Resize image to model input size
        input_shape = self.input_details[0]['shape']
        resized_image = cv2.resize(image, (input_shape[2], input_shape[1]))

        # Normalize image
        normalized_image = (resized_image.astype(np.float32) - 127.5) / 127.5

        # Add batch dimension
        input_data = np.expand_dims(normalized_image, axis=0)

        return input_data

    def run_inference(self, image):
        """Run inference on Coral TPU"""
        input_data = self.preprocess_image(image)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data
```

## ARM-Based Platforms

### Raspberry Pi for Robotics
- **Processor**: ARM Cortex-A72 (Pi 4)
- **Performance**: Limited AI acceleration
- **Cost**: Very affordable
- **Community**: Large ecosystem

### Raspberry Pi AI Acceleration
```python
# Example: Raspberry Pi with AI acceleration
import numpy as np
import cv2
import time

class RaspberryPiAI:
    def __init__(self):
        # Initialize OpenCV DNN for neural network inference
        self.net = cv2.dnn.readNetFromONNX('model.onnx')

        # Performance tracking
        self.fps_history = []

    def preprocess_image(self, image):
        """Preprocess image for Raspberry Pi"""
        # Resize image
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0/255.0,
            size=(224, 224),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        return blob

    def run_inference(self, image):
        """Run inference on Raspberry Pi"""
        blob = self.preprocess_image(image)

        # Set input to the network
        self.net.setInput(blob)

        # Run forward pass
        start_time = time.time()
        output = self.net.forward()
        end_time = time.time()

        inference_time = (end_time - start_time) * 1000  # ms

        return output, inference_time
```

### Odroid Platform
- **Processor**: ARM Cortex-A15, A72, A73
- **Performance**: Good balance of power and performance
- **Use Case**: Intermediate robotics applications

## Model Optimization for Edge Deployment

### Quantization Techniques
```python
# Example: Model quantization for edge deployment
import torch
import torch.quantization as quantization
import torch.nn as nn

class QuantizedModel(nn.Module):
    def __init__(self, original_model):
        super(QuantizedModel, self).__init__()
        self.model = original_model

        # Add quantization layers
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse model layers for better performance"""
        torch.quantization.fuse_modules(self.model, [['conv', 'bn', 'relu']], inplace=True)

def quantize_model(model, dataloader):
    """Quantize model for edge deployment"""
    # Set model to evaluation mode
    model.eval()

    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)

    # Calibrate model with sample data
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            if i >= 10:  # Use first 10 batches for calibration
                break
            model(data)

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    return model
```

### TensorRT Optimization
```python
# Example: TensorRT optimization for NVIDIA platforms
import tensorrt as trt
import numpy as np

def build_tensorrt_engine(onnx_model_path, precision='fp16'):
    """Build TensorRT engine from ONNX model"""
    # Create TensorRT builder
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))

    # Create network definition
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # Parse ONNX model
    parser = trt.OnnxParser(network, trt.Logger())
    with open(onnx_model_path, 'rb') as model_file:
        parser.parse(model_file.read())

    # Configure builder
    config = builder.create_builder_config()

    # Set precision
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)

        # Set up INT8 calibration
        config.int8_calibrator = create_int8_calibrator()

    # Build engine
    engine = builder.build_engine(network, config)

    return engine

def optimize_for_jetson(model_path):
    """Optimize model specifically for Jetson platform"""
    # Load model
    engine = build_tensorrt_engine(model_path, precision='fp16')

    # Serialize engine
    with open('optimized_model.engine', 'wb') as f:
        f.write(engine.serialize())

    return engine
```

## Distributed Computing Architectures

### Multi-Board Architecture
```python
# Example: Distributed computing for humanoid robot
import threading
import multiprocessing
import queue
import time
import numpy as np

class DistributedRobotComputing:
    def __init__(self):
        # Define computing nodes
        self.nodes = {
            'vision': {'board': 'Jetson Nano', 'task': 'perception'},
            'control': {'board': 'Raspberry Pi 4', 'task': 'motion_control'},
            'planning': {'board': 'Jetson Xavier NX', 'task': 'path_planning'}
        }

        # Communication queues
        self.vision_queue = queue.Queue()
        self.control_queue = queue.Queue()
        self.planning_queue = queue.Queue()

        # Start processing threads
        self.start_processing()

    def start_processing(self):
        """Start processing threads for each node"""
        # Vision processing thread
        self.vision_thread = threading.Thread(target=self.vision_processing)
        self.vision_thread.start()

        # Control processing thread
        self.control_thread = threading.Thread(target=self.control_processing)
        self.control_thread.start()

        # Planning processing thread
        self.planning_thread = threading.Thread(target=self.planning_processing)
        self.planning_thread.start()

    def vision_processing(self):
        """Vision processing on dedicated board"""
        while True:
            # Process vision data
            if not self.vision_queue.empty():
                image = self.vision_queue.get()

                # Run object detection
                detections = self.run_object_detection(image)

                # Send to control system
                self.control_queue.put({
                    'detections': detections,
                    'timestamp': time.time()
                })

    def control_processing(self):
        """Motion control processing"""
        while True:
            # Process control commands
            if not self.control_queue.empty():
                data = self.control_queue.get()

                # Generate control commands
                commands = self.generate_control_commands(data)

                # Send to actuators
                self.send_to_actuators(commands)

    def planning_processing(self):
        """Path planning processing"""
        while True:
            # Process planning tasks
            if not self.planning_queue.empty():
                task = self.planning_queue.get()

                # Generate plan
                plan = self.generate_plan(task)

                # Send to control system
                self.control_queue.put({'plan': plan})

    def run_object_detection(self, image):
        """Run object detection on image"""
        # Implementation would use actual detection model
        return [{'class': 'person', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]}]

    def generate_control_commands(self, data):
        """Generate control commands from sensor data"""
        # Implementation would generate actual control commands
        return {'left_leg': 0.1, 'right_leg': 0.1}

    def send_to_actuators(self, commands):
        """Send commands to robot actuators"""
        # Implementation would send to actual actuators
        pass

    def generate_plan(self, task):
        """Generate motion plan for task"""
        # Implementation would generate actual plan
        return {'waypoints': [], 'actions': []}
```

### Edge-Cloud Collaboration
```python
# Example: Edge-cloud collaboration
import requests
import json
import threading
import time

class EdgeCloudCollaboration:
    def __init__(self, edge_processor, cloud_endpoint):
        self.edge_processor = edge_processor
        self.cloud_endpoint = cloud_endpoint
        self.task_queue = queue.Queue()
        self.result_cache = {}

    def process_locally_or_cloud(self, data, task_type):
        """Process task locally or offload to cloud based on complexity"""
        # Determine if task should be processed locally or in cloud
        if self.should_process_locally(data, task_type):
            # Process locally
            result = self.edge_processor.process(data)
        else:
            # Offload to cloud
            result = self.process_on_cloud(data, task_type)

        return result

    def should_process_locally(self, data, task_type):
        """Determine if task should be processed locally"""
        # Criteria for local processing:
        # - Low latency requirements
        # - Privacy concerns
        # - Simple computation
        # - Safety-critical tasks

        if task_type in ['balance_control', 'collision_avoidance']:
            return True  # Safety-critical, process locally
        elif len(data) < 1000000:  # Small data size
            return True  # Process locally to reduce latency
        else:
            return False  # Offload complex tasks to cloud

    def process_on_cloud(self, data, task_type):
        """Process task on cloud"""
        try:
            payload = {
                'data': data.tolist() if isinstance(data, np.ndarray) else data,
                'task_type': task_type,
                'timestamp': time.time()
            }

            response = requests.post(
                f"{self.cloud_endpoint}/process",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                # Fallback to local processing
                return self.edge_processor.process(data)

        except requests.RequestException:
            # Cloud unavailable, process locally
            return self.edge_processor.process(data)
```

## Performance Evaluation and Benchmarking

### Benchmarking Tools
```python
# Example: Performance benchmarking for edge platforms
import time
import psutil
import GPUtil
import numpy as np

class PerformanceBenchmark:
    def __init__(self):
        self.metrics = {
            'inference_time': [],
            'fps': [],
            'cpu_usage': [],
            'gpu_usage': [],
            'memory_usage': [],
            'power_consumption': []  # Estimated
        }

    def benchmark_inference(self, model, test_data, iterations=100):
        """Benchmark inference performance"""
        inference_times = []

        for i in range(iterations):
            start_time = time.time()

            # Run inference
            result = model.run_inference(test_data[i % len(test_data)])

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms
            inference_times.append(inference_time)

            # Collect system metrics
            self.collect_system_metrics()

        # Calculate metrics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        fps = 1000.0 / avg_inference_time  # frames per second

        self.metrics['inference_time'].append({
            'avg': avg_inference_time,
            'std': std_inference_time,
            'min': np.min(inference_times),
            'max': np.max(inference_times),
            'fps': fps
        })

        return {
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'fps': fps
        }

    def collect_system_metrics(self):
        """Collect system resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.metrics['cpu_usage'].append(cpu_percent)

        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        self.metrics['memory_usage'].append(memory_percent)

        # GPU usage (if available)
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_percent = gpus[0].load * 100
            self.metrics['gpu_usage'].append(gpu_percent)
        else:
            self.metrics['gpu_usage'].append(0)

    def compare_platforms(self, platforms, test_data):
        """Compare performance across different platforms"""
        results = {}

        for platform_name, platform in platforms.items():
            print(f"Benchmarking {platform_name}...")

            # Run benchmarks
            inference_result = self.benchmark_inference(platform, test_data)

            # Store results
            results[platform_name] = {
                'inference': inference_result,
                'avg_cpu': np.mean(self.metrics['cpu_usage'][-100:]),
                'avg_memory': np.mean(self.metrics['memory_usage'][-100:]),
                'avg_gpu': np.mean(self.metrics['gpu_usage'][-100:]) if self.metrics['gpu_usage'] else 0
            }

        return results
```

### Power Consumption Analysis
```python
# Example: Power consumption analysis
import subprocess
import time
import matplotlib.pyplot as plt

class PowerAnalyzer:
    def __init__(self):
        self.power_readings = []

    def measure_power_jetson(self):
        """Measure power consumption on Jetson platform"""
        try:
            # Use Jetson stats to measure power
            result = subprocess.run(['jtop', '-c', '1'], capture_output=True, text=True)
            # Parse power information from jtop output
            # This is a simplified example
            power = self.parse_power_from_jtop(result.stdout)
            return power
        except:
            # Fallback to estimation based on utilization
            return self.estimate_power_from_utilization()

    def parse_power_from_jtop(self, jtop_output):
        """Parse power information from jtop output"""
        # Implementation would parse actual jtop output
        # Return power in watts
        return 15.0  # Example value

    def estimate_power_from_utilization(self):
        """Estimate power based on CPU/GPU utilization"""
        cpu_percent = psutil.cpu_percent()
        gpus = GPUtil.getGPUs()
        gpu_percent = gpus[0].load if gpus else 0

        # Power estimation formula (simplified)
        base_power = 5.0  # Base power consumption
        cpu_power = cpu_percent * 0.1  # 0.1W per % CPU
        gpu_power = gpu_percent * 0.2  # 0.2W per % GPU

        estimated_power = base_power + cpu_power + gpu_power
        return estimated_power

    def analyze_power_efficiency(self, models, platform):
        """Analyze power efficiency of different models"""
        efficiency_results = {}

        for model_name, model in models.items():
            # Run model for fixed duration
            start_time = time.time()
            start_power = self.measure_power_jetson()

            # Run inference continuously for 10 seconds
            count = 0
            while time.time() - start_time < 10:
                # Run inference
                dummy_input = np.random.random((1, 3, 224, 224)).astype(np.float32)
                model.run_inference(dummy_input)
                count += 1

            end_time = time.time()
            end_power = self.measure_power_jetson()

            # Calculate metrics
            duration = end_time - start_time
            avg_power = (start_power + end_power) / 2
            inferences_per_second = count / duration
            power_efficiency = inferences_per_second / avg_power  # inferences per watt

            efficiency_results[model_name] = {
                'inferences_per_second': inferences_per_second,
                'avg_power_watts': avg_power,
                'power_efficiency': power_efficiency,
                'total_inferences': count
            }

        return efficiency_results
```

## Platform Selection Guidelines

### Selection Criteria Matrix
```python
class PlatformSelector:
    def __init__(self):
        self.platforms = {
            'Jetson Orin AGX': {
                'performance': 275,  # TOPS
                'power': 60,  # watts
                'cost': 1000,  # USD
                'size': 'medium',
                'ai_features': ['tensor_cores', 'cuda', 'tensorrt']
            },
            'Jetson Nano': {
                'performance': 0.5,
                'power': 10,
                'cost': 100,
                'size': 'small',
                'ai_features': ['cuda', 'tensorrt']
            },
            'Raspberry Pi 4': {
                'performance': 0.01,
                'power': 7,
                'cost': 75,
                'size': 'small',
                'ai_features': ['openvino', 'tensorflow_lite']
            }
        }

    def select_platform(self, requirements):
        """Select platform based on requirements"""
        scores = {}

        for platform_name, specs in self.platforms.items():
            score = 0

            # Performance score (higher is better, normalized)
            perf_score = min(specs['performance'] / requirements.get('min_performance', 1), 10)
            score += perf_score * requirements.get('performance_weight', 0.4)

            # Power score (lower is better if battery powered)
            if requirements.get('battery_powered'):
                power_score = max(0, 10 - (specs['power'] / 5))  # Lower power = higher score
            else:
                power_score = 10 - min(specs['power'] / 20, 10)  # Less penalty if not battery powered
            score += power_score * requirements.get('power_weight', 0.3)

            # Cost score (lower is better)
            cost_score = max(0, 10 - (specs['cost'] / 100))  # Lower cost = higher score
            score += cost_score * requirements.get('cost_weight', 0.3)

            scores[platform_name] = score

        # Return platform with highest score
        best_platform = max(scores, key=scores.get)
        return best_platform, scores[best_platform]
```

## Weekly Breakdown for Chapter 9
- **Week 9.1**: Edge AI platforms overview and characteristics
- **Week 9.2**: NVIDIA Jetson platform and optimization
- **Week 9.3**: Other edge platforms (Intel, Google, ARM)
- **Week 9.4**: Model optimization and deployment strategies

## Assessment
- **Quiz 9.1**: Edge AI platform characteristics and selection (Multiple choice and short answer)
- **Assignment 9.2**: Compare performance of different edge platforms
- **Lab Exercise 9.1**: Optimize and deploy a model on an edge device

## Diagram Placeholders
- ![Edge AI Platform Comparison](./images/edge_ai_platforms_comparison.png)
- ![NVIDIA Jetson Architecture](./images/jetson_architecture.png)
- ![Distributed Computing for Humanoid Robots](./images/distributed_computing_architecture.png)

## Code Snippet: Edge AI Deployment Framework
```python
#!/usr/bin/env python3

import torch
import numpy as np
import time
import json
from abc import ABC, abstractmethod

class EdgeAIDeployer(ABC):
    """Abstract base class for edge AI deployment"""

    def __init__(self, model_path, platform_config):
        self.model_path = model_path
        self.platform_config = platform_config
        self.model = None
        self.is_initialized = False

    @abstractmethod
    def load_model(self):
        """Load model for specific platform"""
        pass

    @abstractmethod
    def preprocess_input(self, input_data):
        """Preprocess input for the model"""
        pass

    @abstractmethod
    def run_inference(self, input_data):
        """Run inference on the model"""
        pass

    def benchmark_performance(self, test_data, iterations=100):
        """Benchmark performance of the deployed model"""
        inference_times = []

        for i in range(iterations):
            preprocessed_data = self.preprocess_input(test_data[i % len(test_data)])

            start_time = time.time()
            result = self.run_inference(preprocessed_data)
            end_time = time.time()

            inference_time = (end_time - start_time) * 1000  # ms
            inference_times.append(inference_time)

        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        performance_metrics = {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time,
            'fps': fps,
            'iterations': iterations
        }

        return performance_metrics

class TensorRTDeployer(EdgeAIDeployer):
    """TensorRT deployment for NVIDIA platforms"""

    def load_model(self):
        """Load TensorRT optimized model"""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        # Load TensorRT engine
        with open(self.model_path, 'rb') as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

        self.is_initialized = True
        print("TensorRT model loaded successfully")

    def preprocess_input(self, input_data):
        """Preprocess input for TensorRT model"""
        # Normalize and reshape input
        if isinstance(input_data, np.ndarray):
            # Ensure correct shape and type
            input_tensor = input_data.astype(np.float32)
            return input_tensor.flatten()
        return input_data

    def run_inference(self, input_data):
        """Run inference using TensorRT"""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data)
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])

        # Run inference
        self.context.execute_v2(bindings=self.bindings)

        # Copy output from device
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])

        # Return output
        output = self.outputs[0]['host'].copy()
        return output

class OpenVINOEdgeDeployer(EdgeAIDeployer):
    """OpenVINO deployment for Intel platforms"""

    def load_model(self):
        """Load OpenVINO optimized model"""
        from openvino.runtime import Core

        self.core = Core()

        # Read model
        self.model = self.core.read_model(model=self.model_path)

        # Compile model
        self.compiled_model = self.core.compile_model(
            model=self.model,
            device_name=self.platform_config.get('device', 'CPU')
        )

        # Get input and output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.is_initialized = True
        print("OpenVINO model loaded successfully")

    def preprocess_input(self, input_data):
        """Preprocess input for OpenVINO model"""
        if isinstance(input_data, np.ndarray):
            # Resize if needed
            input_shape = self.input_layer.shape
            if len(input_shape) == 4:  # NCHW format
                n, c, h, w = input_shape
                if input_data.shape != (n, c, h, w):
                    # Resize and normalize
                    resized = cv2.resize(input_data[0], (w, h))
                    input_data = np.expand_dims(resized.transpose(2, 0, 1), axis=0)

        return input_data

    def run_inference(self, input_data):
        """Run inference using OpenVINO"""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        # Run inference
        result = self.compiled_model([input_data])[self.output_layer]
        return result

class EdgeDeploymentManager:
    """Manager for deploying models to different edge platforms"""

    def __init__(self):
        self.deployers = {
            'tensorrt': TensorRTDeployer,
            'openvino': OpenVINOEdgeDeployer
        }

    def deploy_model(self, model_path, platform_type, platform_config):
        """Deploy model to specified platform"""
        if platform_type not in self.deployers:
            raise ValueError(f"Unsupported platform type: {platform_type}")

        # Create deployer instance
        deployer_class = self.deployers[platform_type]
        deployer = deployer_class(model_path, platform_config)

        # Load model
        deployer.load_model()

        return deployer

    def optimize_for_platform(self, model, platform_type):
        """Optimize model for specific platform"""
        if platform_type == 'tensorrt':
            return self.optimize_for_tensorrt(model)
        elif platform_type == 'openvino':
            return self.optimize_for_openvino(model)
        else:
            return model  # Return as-is for unsupported platforms

    def optimize_for_tensorrt(self, model):
        """Optimize model for TensorRT"""
        import torch
        import torch_tensorrt

        # Convert to TorchScript
        traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

        # Compile with Torch TensorRT
        trt_model = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
            enabled_precisions={torch.float, torch.half}
        )

        return trt_model

    def optimize_for_openvino(self, model):
        """Optimize model for OpenVINO"""
        # Convert to ONNX first, then use OpenVINO model optimizer
        # This is a simplified example
        return model

# Example usage
def main():
    # Initialize deployment manager
    manager = EdgeDeploymentManager()

    # Example platform configurations
    jetson_config = {
        'platform': 'nvidia_jetson',
        'device': 'cuda',
        'precision': 'fp16'
    }

    intel_config = {
        'platform': 'intel_openvino',
        'device': 'CPU',
        'precision': 'fp32'
    }

    # Deploy model to different platforms
    try:
        # Deploy to TensorRT (NVIDIA)
        tensorrt_deployer = manager.deploy_model(
            'path/to/model.plan',
            'tensorrt',
            jetson_config
        )

        # Deploy to OpenVINO (Intel)
        openvino_deployer = manager.deploy_model(
            'path/to/model.xml',
            'openvino',
            intel_config
        )

        # Create test data
        test_data = [np.random.random((1, 3, 224, 224)).astype(np.float32) for _ in range(10)]

        # Benchmark TensorRT deployment
        tensorrt_metrics = tensorrt_deployer.benchmark_performance(test_data)
        print(f"TensorRT Performance: {tensorrt_metrics['fps']:.2f} FPS")

        # Benchmark OpenVINO deployment
        openvino_metrics = openvino_deployer.benchmark_performance(test_data)
        print(f"OpenVINO Performance: {openvino_metrics['fps']:.2f} FPS")

    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    main()
```

## Additional Resources
- NVIDIA Jetson Developer Documentation
- Intel OpenVINO Toolkit Documentation
- Google Coral Documentation
- Raspberry Pi AI Documentation
- Edge AI Benchmarking Tools and Frameworks