---
sidebar_position: 10
title: "Chapter 10: Perception Systems"
---

# Chapter 10: Perception Systems

## Learning Outcomes
By the end of this chapter, students will be able to:
- Design and implement multi-modal perception systems for humanoid robots
- Integrate and calibrate different sensor types (cameras, LIDAR, IMU)
- Apply computer vision techniques for object detection and recognition
- Implement sensor fusion algorithms for enhanced perception
- Evaluate perception system performance in real-world scenarios
- Address challenges in dynamic and uncertain environments

## Overview

Perception systems form the sensory foundation of humanoid robotics, enabling robots to understand and interact with their environment. These systems integrate multiple sensor modalities to provide comprehensive environmental awareness, allowing humanoid robots to navigate, manipulate objects, and interact with humans safely and effectively.

A robust perception system must handle diverse challenges including varying lighting conditions, dynamic environments, occlusions, and sensor noise. For humanoid robots, perception systems must also support social interaction, recognizing human gestures, facial expressions, and intentions.

## Sensor Modalities

### Visual Perception

#### RGB Cameras
RGB cameras provide color information essential for:
- Object recognition and classification
- Scene understanding
- Human detection and tracking
- Visual servoing for manipulation

```python
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class RGBPerception:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.latest_image = None
        self.object_detector = self.initialize_object_detector()

    def image_callback(self, msg):
        """Callback for RGB image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image

            # Process image for perception
            self.process_image(cv_image)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def initialize_object_detector(self):
        """Initialize object detection model"""
        # Using OpenCV DNN module with pre-trained model
        net = cv2.dnn.readNetFromDarknet('config/yolov4.cfg', 'weights/yolov4.weights')
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return {'net': net, 'output_layers': output_layers}

    def process_image(self, image):
        """Process image for object detection"""
        height, width, channels = image.shape

        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.object_detector['net'].setInput(blob)
        outputs = self.object_detector['net'].forward(self.object_detector['output_layers'])

        # Process detection results
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw results
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(class_ids[i])
                confidence = confidences[i]

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{label} {confidence:.2f}",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image, indexes, boxes, confidences, class_ids
```

#### Depth Cameras
Depth cameras provide 3D spatial information:
- Depth estimation for navigation
- 3D object reconstruction
- Safe obstacle avoidance
- Hand-eye coordination for manipulation

```python
import open3d as o3d
import numpy as np

class DepthPerception:
    def __init__(self):
        self.depth_scale = 1000.0  # Scale factor for depth values
        self.camera_intrinsics = None  # To be set from camera info

    def process_depth_image(self, depth_image, rgb_image=None):
        """Process depth image to create point cloud"""
        # Convert depth image to point cloud
        height, width = depth_image.shape

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D coordinates
        z = depth_image / self.depth_scale  # Convert to meters
        x = (u - self.camera_intrinsics[0, 2]) * z / self.camera_intrinsics[0, 0]
        y = (v - self.camera_intrinsics[1, 2]) * z / self.camera_intrinsics[1, 1]

        # Stack coordinates
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # Filter out invalid points (depth = 0)
        valid_points = points[~np.isnan(points).any(axis=1)]
        valid_points = valid_points[valid_points[:, 2] > 0]  # Only positive depths

        return valid_points

    def segment_objects(self, point_cloud, voxel_size=0.01):
        """Segment objects from point cloud using clustering"""
        # Downsample point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Segment plane (ground)
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        # Extract objects (non-ground points)
        object_cloud = pcd_down.select_by_index(inliers, invert=True)

        # Cluster objects using DBSCAN
        labels = np.array(object_cloud.cluster_dbscan(
            eps=0.02,
            min_points=10,
            print_progress=False
        ))

        # Group points by cluster
        unique_labels = set(labels)
        objects = []

        for label in unique_labels:
            if label == -1:  # Noise
                continue

            cluster_indices = np.where(labels == label)[0]
            cluster_points = np.asarray(object_cloud.select_by_index(cluster_indices).points)

            # Calculate object properties
            centroid = np.mean(cluster_points, axis=0)
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(cluster_points)
            )

            objects.append({
                'points': cluster_points,
                'centroid': centroid,
                'bbox': bbox,
                'size': len(cluster_points)
            })

        return objects

    def estimate_surface_normals(self, point_cloud, radius=0.02):
        """Estimate surface normals for point cloud"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=30
            )
        )

        return np.asarray(pcd.normals)
```

### Range Sensors

#### LIDAR Systems
LIDAR provides accurate distance measurements:
- 360-degree environmental mapping
- Precise obstacle detection
- SLAM implementation
- Localization in known environments

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

class LIDARPerception:
    def __init__(self, angle_min=-np.pi, angle_max=np.pi, range_min=0.1, range_max=10.0):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.range_min = range_min
        self.range_max = range_max
        self.angles = None
        self.ranges = None

    def process_scan(self, ranges, angles=None):
        """Process LIDAR scan data"""
        if angles is None:
            # Generate angles if not provided
            num_points = len(ranges)
            angles = np.linspace(self.angle_min, self.angle_max, num_points)

        self.angles = angles
        self.ranges = np.array(ranges)

        # Filter valid ranges
        valid_mask = (self.ranges >= self.range_min) & (self.ranges <= self.range_max)
        valid_angles = self.angles[valid_mask]
        valid_ranges = self.ranges[valid_mask]

        # Convert to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)

        points = np.column_stack((x, y))

        return points

    def detect_obstacles(self, points, min_cluster_size=5):
        """Detect obstacles using clustering"""
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size)
        cluster_labels = clustering.fit_predict(points)

        obstacles = []
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            cluster_mask = cluster_labels == label
            cluster_points = points[cluster_mask]

            # Calculate obstacle properties
            centroid = np.mean(cluster_points, axis=0)
            bbox = [
                np.min(cluster_points[:, 0]), np.min(cluster_points[:, 1]),
                np.max(cluster_points[:, 0]), np.max(cluster_points[:, 1])
            ]

            obstacles.append({
                'points': cluster_points,
                'centroid': centroid,
                'bbox': bbox,
                'size': len(cluster_points)
            })

        return obstacles

    def create_occupancy_grid(self, points, grid_size=100, resolution=0.1):
        """Create occupancy grid from LIDAR points"""
        # Determine grid bounds
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        # Create grid
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Convert points to grid coordinates
        x_coords = ((points[:, 0] - x_min) / resolution).astype(int)
        y_coords = ((points[:, 1] - y_min) / resolution).astype(int)

        # Mark occupied cells
        valid_mask = (x_coords >= 0) & (x_coords < grid_size) & \
                     (y_coords >= 0) & (y_coords < grid_size)

        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]

        grid[y_valid, x_valid] = 255  # Occupied

        return grid, (x_min, y_min, resolution)

    def detect_free_space(self, points, robot_radius=0.3):
        """Detect free space around robot"""
        # Calculate distances from robot (assumed at origin)
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)

        # Free space is area beyond obstacle points
        free_space_mask = distances > robot_radius

        return points[free_space_mask]
```

#### Ultrasonic Sensors
Ultrasonic sensors provide short-range detection:
- Close-range obstacle detection
- Wall following
- Simple navigation tasks
- Complementary to other sensors

### Inertial Sensors

#### IMU Integration
IMUs provide orientation and motion data:
- Robot pose estimation
- Balance control feedback
- Motion tracking
- Drift correction for other sensors

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import Imu

class IMUPerception:
    def __init__(self):
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # State variables
        self.orientation = np.array([0, 0, 0, 1])  # quaternion [x, y, z, w]
        self.angular_velocity = np.array([0, 0, 0])
        self.linear_acceleration = np.array([0, 0, 0])

        # Integration parameters
        self.last_time = rospy.Time.now()
        self.bias = np.array([0, 0, 0])  # Gyro bias
        self.gravity = 9.81

    def imu_callback(self, msg):
        """Process IMU data"""
        # Extract orientation (if available)
        if msg.orientation.w != 0 or msg.orientation.x != 0 or msg.orientation.y != 0 or msg.orientation.z != 0:
            self.orientation = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])

        # Extract angular velocity
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Extract linear acceleration
        self.linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Calculate time delta
        current_time = msg.header.stamp
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Update state using integration
        if dt > 0:
            self.update_orientation(dt)

    def update_orientation(self, dt):
        """Update orientation using gyroscope integration"""
        # Remove bias from angular velocity
        corrected_angular_velocity = self.angular_velocity - self.bias

        # Convert to rotation vector
        rotation_vector = corrected_angular_velocity * dt

        # Convert to quaternion
        rotation = R.from_rotvec(rotation_vector)
        rotation_quat = rotation.as_quat()  # [x, y, z, w]

        # Integrate with current orientation
        current_rotation = R.from_quat(self.orientation)
        new_rotation = current_rotation * R.from_quat(rotation_quat)

        self.orientation = new_rotation.as_quat()

    def get_euler_angles(self):
        """Convert orientation quaternion to Euler angles"""
        rotation = R.from_quat(self.orientation)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        return euler_angles

    def get_roll_pitch(self):
        """Get roll and pitch angles for balance control"""
        euler = self.get_euler_angles()
        roll, pitch, _ = euler
        return roll, pitch

    def calibrate_gyro(self, calibration_samples=100):
        """Calibrate gyroscope bias"""
        samples = []
        for _ in range(calibration_samples):
            samples.append(self.angular_velocity.copy())
            rospy.sleep(0.01)

        # Calculate bias as average of samples when robot is stationary
        self.bias = np.mean(samples, axis=0)
        rospy.loginfo(f"IMU Gyro bias calibrated: {self.bias}")
```

## Sensor Fusion

### Kalman Filter Implementation
```python
import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector [x, y, vx, vy]
        self.x = np.zeros(state_dim)

        # State covariance matrix
        self.P = np.eye(state_dim) * 1000

        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 1.0

        # State transition matrix (constant velocity model)
        self.F = np.eye(state_dim)
        self.F[0, 2] = 1  # x = x + vx*dt
        self.F[1, 3] = 1  # y = y + vy*dt

        # Measurement matrix
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[0, 0] = 1  # Measure x
        self.H[1, 1] = 1  # Measure y

    def predict(self, dt=1.0):
        """Predict next state"""
        # Update state transition matrix with time
        F = self.F.copy()
        F[0, 2] = dt
        F[1, 3] = dt

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update state with measurement"""
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Calculate innovation
        y = measurement - self.H @ self.x

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

class SensorFusion:
    def __init__(self):
        # Initialize Kalman filter for position tracking
        self.kf = KalmanFilter(state_dim=4, measurement_dim=2)  # [x, y, vx, vy], measure [x, y]

        # Sensor data storage
        self.camera_measurements = []
        self.lidar_measurements = []
        self.odom_measurements = []

        # Timestamps
        self.last_update = rospy.Time.now()

    def add_camera_measurement(self, x, y, timestamp):
        """Add camera-based position measurement"""
        self.camera_measurements.append((x, y, timestamp))

    def add_lidar_measurement(self, x, y, timestamp):
        """Add LIDAR-based position measurement"""
        self.lidar_measurements.append((x, y, timestamp))

    def add_odom_measurement(self, x, y, timestamp):
        """Add odometry-based position measurement"""
        self.odom_measurements.append((x, y, timestamp))

    def fused_position_estimate(self):
        """Estimate fused position from all sensors"""
        # Get current time
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update).to_sec()
        self.last_update = current_time

        # Predict state
        self.kf.predict(dt)

        # Fuse measurements from different sensors
        # Weight measurements based on sensor reliability
        measurements = []

        # Process camera measurements
        if self.camera_measurements:
            cam_x, cam_y, _ = self.camera_measurements[-1]  # Use most recent
            measurements.append((np.array([cam_x, cam_y]), 0.5))  # Lower weight

        # Process LIDAR measurements
        if self.lidar_measurements:
            lidar_x, lidar_y, _ = self.lidar_measurements[-1]
            measurements.append((np.array([lidar_x, lidar_y]), 1.0))  # Higher weight

        # Process odometry measurements
        if self.odom_measurements:
            odom_x, odom_y, _ = self.odom_measurements[-1]
            measurements.append((np.array([odom_x, odom_y]), 0.8))  # Medium weight

        # Update filter with weighted measurements
        for measurement, weight in measurements:
            # Adjust measurement noise based on weight
            weighted_R = self.kf.R / weight
            original_R = self.kf.R.copy()
            self.kf.R = weighted_R
            self.kf.update(measurement)
            self.kf.R = original_R  # Restore original noise

        # Return estimated position
        return self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]  # x, y, vx, vy
```

### Particle Filter for Non-linear Systems
```python
class ParticleFilter:
    def __init__(self, num_particles=1000, state_dim=4):
        self.num_particles = num_particles
        self.state_dim = state_dim

        # Initialize particles randomly
        self.particles = np.random.normal(0, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input, process_noise=0.1):
        """Predict particle states based on control input"""
        # Add process noise
        noise = np.random.normal(0, process_noise, self.particles.shape)

        # Simple motion model (could be more complex)
        self.particles += control_input + noise

    def update(self, measurement, measurement_noise=0.1):
        """Update particle weights based on measurement"""
        # Calculate likelihood of each particle given measurement
        diff = self.particles - measurement
        likelihood = np.exp(-0.5 * np.sum(diff**2, axis=1) / measurement_noise**2)

        # Update weights
        self.weights *= likelihood

        # Normalize weights
        self.weights += 1e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = []
        cumulative_sum = np.cumsum(self.weights)
        start = np.random.uniform(0, 1/self.num_particles)

        for i in range(self.num_particles):
            index = 0
            threshold = start + i / self.num_particles
            while cumulative_sum[index] < threshold:
                index += 1
            indices.append(index)

        # Resample particles and reset weights
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """Get state estimate as weighted average of particles"""
        return np.average(self.particles, axis=0, weights=self.weights)
```

## Computer Vision for Robotics

### Feature Detection and Matching
```python
import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, detector_type='SIFT'):
        self.detector_type = detector_type
        self.detector = self.initialize_detector(detector_type)

    def initialize_detector(self, detector_type):
        """Initialize feature detector"""
        if detector_type == 'SIFT':
            return cv2.SIFT_create()
        elif detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=1000)
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")

    def detect_features(self, image):
        """Detect features in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_threshold=0.7):
        """Match features between two images"""
        # Use FLANN matcher for efficiency
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_pose(self, img1, img2):
        """Estimate relative pose between two images"""
        # Detect features in both images
        kp1, desc1 = self.detect_features(img1)
        kp2, desc2 = self.detect_features(img2)

        if desc1 is None or desc2 is None:
            return None, None

        # Match features
        matches = self.match_features(desc1, desc2)

        if len(matches) >= 10:  # Need minimum matches for pose estimation
            # Get matched keypoint coordinates
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography matrix
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            return H, matches
        else:
            return None, []
```

### Object Recognition and Classification
```python
import torch
import torchvision.transforms as transforms
from PIL import Image

class ObjectClassifier:
    def __init__(self, model_path=None, model_type='resnet'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        if model_type == 'resnet':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        elif model_type == 'mobilenet':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        # Load ImageNet class labels
        self.classes = self.load_imagenet_classes()

    def load_imagenet_classes(self):
        """Load ImageNet class labels"""
        # In practice, you would load the actual ImageNet class file
        # For this example, we'll return a simple placeholder
        return [f"class_{i}" for i in range(1000)]

    def classify_image(self, image_path_or_array):
        """Classify an image"""
        if isinstance(image_path_or_array, str):
            # Load image from file
            image = Image.open(image_path_or_array).convert('RGB')
        else:
            # Convert numpy array to PIL image
            image = Image.fromarray(image_path_or_array.astype('uint8'), 'RGB')

        # Preprocess image
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Get top predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        results = []
        for i in range(5):
            class_id = top5_catid[i].item()
            probability = top5_prob[i].item()
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"unknown_{class_id}"

            results.append({
                'class_id': class_id,
                'class_name': class_name,
                'probability': probability
            })

        return results

class ObjectDetector:
    def __init__(self, model_type='yolov5'):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained object detection model
        if model_type == 'yolov5':
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif model_type == 'fasterrcnn':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)

        self.model = self.model.to(self.device)
        self.model.eval()

    def detect_objects(self, image_path_or_array):
        """Detect objects in an image"""
        if isinstance(image_path_or_array, str):
            # Load image from file
            image = Image.open(image_path_or_array).convert('RGB')
        else:
            # Convert numpy array to PIL image
            image = Image.fromarray(image_path_or_array.astype('uint8'), 'RGB')

        # Perform object detection
        results = self.model([image])

        # Extract detection results
        detections = []
        for result in results:
            boxes = result['boxes'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            scores = result['scores'].cpu().numpy()

            for i in range(len(boxes)):
                if scores[i] > 0.5:  # Confidence threshold
                    detection = {
                        'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                        'label': int(labels[i]),
                        'confidence': float(scores[i]),
                        'class_name': self.get_coco_class_name(int(labels[i]))
                    }
                    detections.append(detection)

        return detections

    def get_coco_class_name(self, class_id):
        """Get COCO dataset class name for class ID"""
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        if class_id < len(coco_names):
            return coco_names[class_id]
        else:
            return f"unknown_{class_id}"
```

## Human Perception

### Face Detection and Recognition
```python
import cv2
import numpy as np
import face_recognition

class HumanPerception:
    def __init__(self):
        # Initialize face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Initialize face recognition encodings
        self.known_face_encodings = []
        self.known_face_names = []

    def detect_faces(self, image):
        """Detect faces in image using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                'bbox': [x, y, x+w, y+h],
                'center': [x + w//2, y + h//2],
                'size': w * h
            })

        return face_data

    def recognize_faces(self, image):
        """Recognize known faces in image"""
        # Convert image to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        face_data = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find best match
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_data.append({
                'bbox': [left, top, right, bottom],
                'name': name,
                'confidence': 1 - face_distances[best_match_index] if len(face_distances) > 0 else 0
            })

        return face_data

    def add_known_person(self, image, name):
        """Add a known person to the recognition database"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)

        if encodings:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
            return True
        return False

    def detect_gestures(self, hand_image):
        """Detect hand gestures for human-robot interaction"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(hand_image, cv2.COLOR_BGR2HSV)

        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (assumed to be hand)
            hand_contour = max(contours, key=cv2.contourArea)

            # Calculate convex hull
            hull = cv2.convexHull(hand_contour, returnPoints=False)

            # Find convexity defects
            defects = cv2.convexityDefects(hand_contour, hull)

            if defects is not None:
                # Count defects (gaps between fingers)
                defect_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])

                    # Calculate angles and distances
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                    # Apply cosine rule to find angle
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                    # If angle is less than 90, it's a finger gap
                    if angle <= 90:
                        defect_count += 1

                gesture = self.interpret_gesture(defect_count)
                return gesture, hand_contour

        return "unknown", None

    def interpret_gesture(self, defect_count):
        """Interpret gesture based on number of defects"""
        if defect_count == 0:
            return "fist"  # 1 finger up (thumb)
        elif defect_count == 1:
            return "one"   # 2 fingers up
        elif defect_count == 2:
            return "two"   # 3 fingers up
        elif defect_count == 3:
            return "three" # 4 fingers up
        elif defect_count == 4:
            return "five"  # 5 fingers up (palm)
        else:
            return "unknown"
```

## Perception Pipeline for Humanoid Robots

### Integrated Perception System
```python
import threading
import queue
import time
from collections import deque

class IntegratedPerceptionSystem:
    def __init__(self):
        # Initialize sensor modules
        self.rgb_perception = RGBPerception()
        self.depth_perception = DepthPerception()
        self.lidar_perception = LIDARPerception()
        self.imu_perception = IMUPerception()
        self.human_perception = HumanPerception()

        # Initialize fusion module
        self.sensor_fusion = SensorFusion()

        # Data queues
        self.perception_queue = queue.Queue(maxsize=10)
        self.navigation_queue = queue.Queue(maxsize=10)

        # Thread control
        self.running = True
        self.perception_thread = threading.Thread(target=self.perception_loop)

        # Performance tracking
        self.fps_counter = deque(maxlen=30)  # Last 30 frames

        # Start perception thread
        self.perception_thread.start()

    def perception_loop(self):
        """Main perception processing loop"""
        frame_count = 0
        start_time = time.time()

        while self.running:
            # Process perception data
            self.process_sensors()

            # Fuse sensor data
            fused_data = self.fuse_sensor_data()

            # Publish perception results
            if not self.perception_queue.full():
                self.perception_queue.put(fused_data)

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps = frame_count
                self.fps_counter.append(fps)
                frame_count = 0
                start_time = current_time

                # Log performance
                rospy.loginfo(f"Perception FPS: {fps}, Avg: {np.mean(self.fps_counter):.1f}")

            # Small delay to prevent busy waiting
            time.sleep(0.01)

    def process_sensors(self):
        """Process data from all sensors"""
        # Process RGB camera
        if self.rgb_perception.latest_image is not None:
            rgb_image = self.rgb_perception.latest_image
            objects, humans = self.process_vision_data(rgb_image)

            # Update fusion system
            # self.sensor_fusion.add_camera_measurement(obj_x, obj_y, rospy.Time.now())

        # Process depth camera
        # Process LIDAR
        # Process IMU

        pass  # Implementation would continue with other sensors

    def process_vision_data(self, image):
        """Process vision data for objects and humans"""
        # Detect objects
        object_results, _, _, _, _ = self.rgb_perception.process_image(image)

        # Detect humans/faces
        human_results = self.human_perception.recognize_faces(image)

        return object_results, human_results

    def fuse_sensor_data(self):
        """Fuse data from multiple sensors"""
        # Get current sensor estimates
        position_estimate = self.sensor_fusion.fused_position_estimate()

        # Combine with other sensor data
        fused_data = {
            'position': position_estimate[:2],  # x, y
            'velocity': position_estimate[2:],  # vx, vy
            'orientation': self.imu_perception.get_euler_angles(),
            'objects': [],  # Would be populated from vision processing
            'humans': [],   # Would be populated from human detection
            'timestamp': rospy.Time.now()
        }

        return fused_data

    def get_perception_data(self):
        """Get latest perception data"""
        try:
            if not self.perception_queue.empty():
                return self.perception_queue.get_nowait()
        except queue.Empty:
            pass
        return None

    def stop(self):
        """Stop perception system"""
        self.running = False
        self.perception_thread.join()

# Example usage
def main():
    # Initialize perception system
    perception_system = IntegratedPerceptionSystem()

    try:
        # Main loop
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            # Get perception data
            data = perception_system.get_perception_data()

            if data:
                # Use perception data for navigation, manipulation, etc.
                rospy.loginfo(f"Perceived position: {data['position']}")

                # Example: Check for humans in environment
                if data['humans']:
                    rospy.loginfo(f"Detected {len(data['humans'])} humans")

            rate.sleep()

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down perception system")
    finally:
        perception_system.stop()

if __name__ == '__main__':
    rospy.init_node('perception_system')
    main()
```

## Weekly Breakdown for Chapter 10
- **Week 10.1**: Sensor modalities and integration
- **Week 10.2**: Computer vision techniques for robotics
- **Week 10.3**: Sensor fusion algorithms and implementation
- **Week 10.4**: Human perception and interaction systems

## Assessment
- **Quiz 10.1**: Sensor modalities and their applications (Multiple choice and short answer)
- **Assignment 10.2**: Implement a sensor fusion algorithm
- **Lab Exercise 10.1**: Develop a computer vision pipeline for object detection

## Diagram Placeholders
- ![Perception System Architecture](./images/perception_system_architecture.png)
- ![Sensor Fusion Pipeline](./images/sensor_fusion_pipeline.png)
- ![Human-Robot Interaction Perception](./images/human_robot_interaction_perception.png)

## Code Snippet: Complete Perception System
```python
#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, LaserScan, Imu
from cv_bridge import CvBridge
import threading
import queue
import time
from collections import deque

class CompletePerceptionSystem:
    """
    Complete perception system for humanoid robots integrating
    multiple sensors and processing modalities
    """
    def __init__(self):
        # Initialize ROS components
        self.bridge = CvBridge()

        # Initialize sensor data storage
        self.rgb_image = None
        self.depth_image = None
        self.laser_scan = None
        self.imu_data = None

        # Initialize processing modules
        self.object_detector = ObjectDetector(model_type='yolov5')
        self.human_detector = HumanPerception()
        self.lidar_processor = LIDARPerception()
        self.kalman_filter = KalmanFilter(state_dim=6, measurement_dim=3)  # x, y, z, vx, vy, vz

        # Data queues for multi-threading
        self.perception_queue = queue.Queue(maxsize=20)
        self.visualization_queue = queue.Queue(maxsize=10)

        # Performance tracking
        self.frame_times = deque(maxlen=50)
        self.detection_times = deque(maxlen=50)

        # ROS subscribers
        self.rgb_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # Processing control
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.start()

        rospy.loginfo("Complete Perception System initialized")

    def rgb_callback(self, msg):
        """Handle RGB camera data"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error processing RGB image: {e}")

    def depth_callback(self, msg):
        """Handle depth camera data"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

    def scan_callback(self, msg):
        """Handle LIDAR scan data"""
        self.laser_scan = msg

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg

    def processing_loop(self):
        """Main processing loop"""
        while self.running:
            start_time = time.time()

            # Process all available sensor data
            perception_result = self.process_perception()

            # Add processing time to performance tracking
            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)

            # Publish result if queue not full
            if not self.perception_queue.full() and perception_result:
                self.perception_queue.put(perception_result)

            # Small delay to prevent busy waiting
            time.sleep(0.01)

    def process_perception(self):
        """Process all sensor data and generate perception results"""
        if self.rgb_image is None:
            return None

        start_time = time.time()

        # Process RGB image for objects and humans
        rgb_result = self.process_rgb_perception()

        # Process depth image for 3D information
        depth_result = self.process_depth_perception()

        # Process LIDAR for environment mapping
        lidar_result = self.process_lidar_perception()

        # Process IMU for orientation
        imu_result = self.process_imu_perception()

        # Fuse all sensor data
        fused_result = self.fuse_sensor_data(rgb_result, depth_result, lidar_result, imu_result)

        # Add processing time to tracking
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)

        return {
            'timestamp': rospy.Time.now(),
            'objects': fused_result.get('objects', []),
            'humans': fused_result.get('humans', []),
            'environment': fused_result.get('environment', {}),
            'position': fused_result.get('position', [0, 0, 0]),
            'orientation': fused_result.get('orientation', [0, 0, 0]),
            'processing_time': detection_time,
            'performance': {
                'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
                'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
                'fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0
            }
        }

    def process_rgb_perception(self):
        """Process RGB camera data"""
        if self.rgb_image is None:
            return {'objects': [], 'humans': []}

        # Detect objects
        objects = self.object_detector.detect_objects(self.rgb_image)

        # Detect humans
        humans = self.human_detector.recognize_faces(self.rgb_image)

        return {
            'objects': objects,
            'humans': humans,
            'image_shape': self.rgb_image.shape
        }

    def process_depth_perception(self):
        """Process depth camera data"""
        if self.depth_image is None:
            return {'depth_map': None, 'point_cloud': None}

        # Convert depth to point cloud
        height, width = self.depth_image.shape
        # Create simple point cloud from depth
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        depth_values = self.depth_image

        # Simple conversion (would need proper camera intrinsics in practice)
        x = (u - width/2) * depth_values / 1000  # Simplified
        y = (v - height/2) * depth_values / 1000
        z = depth_values

        point_cloud = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        valid_points = point_cloud[~np.isnan(point_cloud).any(axis=1)]
        valid_points = valid_points[valid_points[:, 2] > 0]  # Only positive depths

        return {
            'depth_map': self.depth_image,
            'point_cloud': valid_points
        }

    def process_lidar_perception(self):
        """Process LIDAR data"""
        if self.laser_scan is None:
            return {'obstacles': [], 'free_space': []}

        # Convert scan to Cartesian points
        angles = np.linspace(self.laser_scan.angle_min, self.laser_scan.angle_max,
                           len(self.laser_scan.ranges))
        ranges = np.array(self.laser_scan.ranges)

        # Filter valid ranges
        valid_mask = (ranges >= self.laser_scan.range_min) & (ranges <= self.laser_scan.range_max)
        valid_angles = angles[valid_mask]
        valid_ranges = ranges[valid_mask]

        # Convert to Cartesian
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        points = np.column_stack((x, y))

        # Detect obstacles using clustering
        obstacles = self.lidar_processor.detect_obstacles(points)

        return {
            'points': points,
            'obstacles': obstacles,
            'free_space': self.lidar_processor.detect_free_space(points)
        }

    def process_imu_perception(self):
        """Process IMU data"""
        if self.imu_data is None:
            return {'orientation': [0, 0, 0], 'angular_velocity': [0, 0, 0]}

        # Extract orientation quaternion
        quat = [self.imu_data.orientation.x, self.imu_data.orientation.y,
                self.imu_data.orientation.z, self.imu_data.orientation.w]

        # Convert to Euler angles
        import tf.transformations as tf_trans
        euler = tf_trans.euler_from_quaternion(quat)

        return {
            'orientation': list(euler),
            'angular_velocity': [
                self.imu_data.angular_velocity.x,
                self.imu_data.angular_velocity.y,
                self.imu_data.angular_velocity.z
            ]
        }

    def fuse_sensor_data(self, rgb_result, depth_result, lidar_result, imu_result):
        """Fuse data from all sensors"""
        fused_data = {
            'objects': [],
            'humans': [],
            'environment': {},
            'position': [0, 0, 0],
            'orientation': [0, 0, 0]
        }

        # Fuse object detections with depth information
        if rgb_result and depth_result:
            # Associate 2D detections with 3D points
            for obj in rgb_result.get('objects', []):
                # Convert 2D bounding box center to 3D position using depth
                bbox = obj['bbox']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)

                if (0 <= center_x < depth_result['depth_map'].shape[1] and
                    0 <= center_y < depth_result['depth_map'].shape[0]):
                    depth = depth_result['depth_map'][center_y, center_x]

                    # Convert to 3D coordinates (simplified)
                    obj['position_3d'] = [
                        (center_x - depth_result['depth_map'].shape[1]/2) * depth / 1000,
                        (center_y - depth_result['depth_map'].shape[0]/2) * depth / 1000,
                        depth
                    ]

                fused_data['objects'].append(obj)

        # Add human detections
        fused_data['humans'] = rgb_result.get('humans', []) if rgb_result else []

        # Incorporate LIDAR environment mapping
        fused_data['environment'] = {
            'obstacles': lidar_result.get('obstacles', []),
            'free_space': lidar_result.get('free_space', [])
        }

        # Use IMU for orientation
        fused_data['orientation'] = imu_result.get('orientation', [0, 0, 0])

        # Estimate position using sensor fusion (simplified)
        # In practice, this would use more sophisticated fusion algorithms
        fused_data['position'] = [0, 0, 0]  # Would be estimated from multiple sensors

        return fused_data

    def get_perception_result(self):
        """Get latest perception result"""
        try:
            if not self.perception_queue.empty():
                return self.perception_queue.get_nowait()
        except queue.Empty:
            pass
        return None

    def visualize_perception(self, image, perception_result):
        """Visualize perception results on image"""
        if image is None or perception_result is None:
            return image

        vis_image = image.copy()

        # Draw object detections
        for obj in perception_result.get('objects', []):
            if 'bbox' in obj:
                bbox = [int(x) for x in obj['bbox']]
                cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(vis_image, f"{obj['class_name']} {obj['confidence']:.2f}",
                           (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw human detections
        for human in perception_result.get('humans', []):
            if 'bbox' in human:
                bbox = [int(x) for x in human['bbox']]
                cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(vis_image, human['name'], (bbox[0], bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add performance info
        perf = perception_result.get('performance', {})
        cv2.putText(vis_image, f"FPS: {perf.get('fps', 0):.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Proc Time: {perf.get('avg_detection_time', 0)*1000:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_image

    def stop(self):
        """Stop the perception system"""
        self.running = False
        self.processing_thread.join()

def main():
    """Main function to run the perception system"""
    rospy.init_node('complete_perception_system')

    # Initialize perception system
    perception_system = CompletePerceptionSystem()

    # Create display window
    cv2.namedWindow('Perception Visualization', cv2.WINDOW_AUTOSIZE)

    try:
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            # Get perception result
            result = perception_system.get_perception_result()

            # Visualize if we have an RGB image and results
            if perception_system.rgb_image is not None and result:
                vis_image = perception_system.visualize_perception(
                    perception_system.rgb_image, result)
                cv2.imshow('Perception Visualization', vis_image)

            # Handle window events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            rate.sleep()

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down perception system")
    finally:
        perception_system.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

## Additional Resources
- OpenCV Documentation for Computer Vision
- Point Cloud Library (PCL) for 3D Processing
- Robot Operating System (ROS) Perception Tutorials
- Computer Vision and Pattern Recognition Resources
- Sensor Fusion and State Estimation Techniques