# Semantic Mapper VSLAM

**RTAB-Map VSLAM + YOLO 객체 검출을 이용한 정확한 3D 시맨틱 매핑**

RealSense D455 카메라와 Agile Tracer 로봇을 사용하여 실시간으로 객체를 검출하고, RTAB-Map의 포인트클라우드 맵과 융합하여 정확한 위치, 크기, 방향을 가진 3D 객체를 생성합니다.

---

## 시스템 구성

```
┌─────────────────────┐
│  RealSense D455     │
│  RGB-D Camera       │
└──────┬──────────────┘
       │
       ├─> RGB + Depth + IMU
       │
       v
┌──────────────────────────┐
│  RTAB-Map VSLAM         │
│  Visual SLAM             │
│  - Odometry             │
│  - Loop Closure         │
│  - 3D Map               │
└──────┬───────────────────┘
       │
       ├─> /rtabmap/cloud_map
       │   /rtabmap/odom
       │
       v
┌──────────────────────────┐
│  YOLO Depth Mapper      │
│  - YOLOv8 TensorRT      │
│  - 3D Position          │
│  - Persistence Filter   │
└──────┬───────────────────┘
       │
       v /semantic_mapper/detections
       │
┌──────┴───────────────────┐
│  RTAB-Map Fusion        │
│  - Size Refinement      │
│  - Position Correction  │
│  - Orientation (PCA)    │
└──────┬───────────────────┘
       │
       v /semantic_mapper/detections_fused
       │
       v
  [가상맵 객체 생성]
```

---

## 주요 기능

### 1. YOLO Depth Mapper
- **YOLOv8 TensorRT** 엔진을 사용한 실시간 객체 검출
- RealSense D455 depth 정보로 **3D 위치** 추정
- **Persistence tracking**: 3회 연속 검출 후 확정 (노이즈 제거)
- **EMA 필터링**: 위치 떨림 최소화
- **13개 클래스 필터**: 실내 가구/전자기기만 검출
  - tv, cup, monitor, laptop, chair, couch, book, keyboard, mouse, potted plant, bottle, cell phone, dining table

### 2. RTAB-Map Fusion
- RTAB-Map 포인트클라우드 맵과 융합
- **PCA 기반 크기 추정**: 실제 객체 크기 계산
- **위치 보정**: 포인트 클러스터 중심으로 위치 조정
- **방향 추정**: PCA 주성분 분석으로 yaw 각도 계산
- **정확성 향상**: ws_slam 대비 훨씬 높은 정확도

---

## 설치 및 빌드

### 1. 의존성 설치

```bash
# ROS 2 Humble 기반
sudo apt install -y \
    ros-humble-realsense2-camera \
    ros-humble-rtabmap-ros \
    ros-humble-imu-filter-madgwick \
    python3-ultralytics \
    python3-opencv \
    python3-torch
```

### 2. YOLO 모델 준비

```bash
# YOLOv8l TensorRT 엔진 변환 (Jetson Orin NX에서)
mkdir -p ~/models
cd ~/models

# PyTorch 모델 다운로드
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt

# TensorRT 엔진으로 변환 (640x640 input, FP16)
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov8l.pt')
model.export(format='engine', imgsz=640, half=True, device=0)
EOF

# 생성된 파일: yolov8l.engine
```

### 3. 빌드

```bash
cd ~/vslam_ws
colcon build --packages-select semantic_mapper_msgs semantic_mapper_vslam --symlink-install
source install/setup.bash
```

---

## 사용 방법

### 전체 시스템 실행

```bash
# vslam_ws 워크스페이스 소싱
source ~/vslam_ws/install/setup.bash

# 전체 시스템 실행 (RealSense + RTAB-Map + YOLO)
ros2 launch semantic_mapper_vslam semantic_vslam_complete.launch.py \
    model_path:=~/models/yolov8l.engine \
    yolo_conf:=0.35 \
    yolo_input_size:=640
```

### 개별 실행

#### 1) RealSense D455만 실행
```bash
ros2 launch semantic_mapper_vslam realsense_d455.launch.py
```

#### 2) RTAB-Map VSLAM만 실행
```bash
ros2 launch semantic_mapper_vslam rtabmap_vslam.launch.py
```

#### 3) YOLO + Fusion만 실행 (RealSense와 RTAB-Map이 이미 실행 중일 때)
```bash
# YOLO Depth Mapper
ros2 run semantic_mapper_vslam yolo_depth_mapper \
    --ros-args \
    -p model_path:=~/models/yolov8l.engine \
    -p conf_threshold:=0.35

# RTAB-Map Fusion (별도 터미널)
ros2 run semantic_mapper_vslam yolo_rtabmap_fusion
```

---

## 파라미터 설정

### YOLO Depth Mapper 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `model_path` | `yolov8l.engine` | YOLO 모델 경로 (.engine 또는 .pt) |
| `conf_threshold` | `0.35` | 검출 신뢰도 임계값 |
| `iou_threshold` | `0.50` | NMS IoU 임계값 |
| `input_size` | `640` | YOLO 입력 크기 (640, 896 등) |
| `device` | `cuda:0` | 추론 장치 (cuda:0 또는 cpu) |
| `min_box_px` | `30` | 최소 박스 크기 (픽셀) |
| `core_roi` | `0.60` | 깊이 추출 ROI 비율 (중심부 60%) |
| `persist_k` | `3` | 확정까지 필요한 연속 검출 횟수 |
| `persist_t` | `10` | 삭제까지 허용 miss 횟수 |
| `ema_alpha` | `0.40` | EMA 필터 계수 (낮을수록 부드러움) |
| `match_dist_m` | `0.40` | 트래킹 매칭 거리 (m) |
| `max_fps` | `10.0` | 최대 처리 속도 (Hz) |

### RTAB-Map Fusion 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `search_radius` | `0.7` | 객체 주변 포인트 탐색 반경 (m) |
| `downsample_stride` | `4` | 포인트클라우드 다운샘플링 stride |
| `min_points` | `50` | PCA 계산 최소 포인트 개수 |
| `percentile` | `0.95` | 로버스트 크기 추정 백분위수 |
| `adjust_position` | `true` | 위치 보정 활성화 |
| `max_shift_m` | `0.5` | 최대 위치 이동 거리 (m) |
| `estimate_orientation` | `true` | 방향 추정 활성화 |
| `max_fps` | `10.0` | 최대 처리 속도 (Hz) |

---

## 토픽 구조

### 입력 토픽
- `/camera/camera/color/image_raw` - RGB 이미지
- `/camera/camera/aligned_depth_to_color/image_raw` - 정렬된 Depth 이미지
- `/camera/camera/color/camera_info` - 카메라 내부 파라미터
- `/camera/camera/imu` - IMU 데이터
- `/rtabmap/cloud_map` - RTAB-Map 포인트클라우드 맵

### 출력 토픽
- `/semantic_mapper/detections` - 원본 YOLO 검출 결과 (3D)
- `/semantic_mapper/detections_fused` - RTAB-Map과 융합된 최종 결과 (정확한 위치/크기/방향)
- `/yolo/debug_image` - 시각화 이미지 (바운딩 박스)
- `/yolo/marker_array` - RViz 마커 (3D 박스)

### TF Frames
- `map` - RTAB-Map 글로벌 맵 프레임
- `odom` - RTAB-Map odometry 프레임
- `camera_link` - 카메라 베이스 프레임
- `camera_color_optical_frame` - RGB 카메라 광학 프레임

---

## RViz2 시각화

```bash
# RViz2 실행
rviz2

# 다음 항목 추가:
# 1. Fixed Frame: map
# 2. Add -> Image -> /yolo/debug_image (YOLO 검출 결과)
# 3. Add -> MarkerArray -> /yolo/marker_array (3D 바운딩 박스)
# 4. Add -> PointCloud2 -> /rtabmap/cloud_map (RTAB-Map 맵)
# 5. Add -> TF (좌표계 표시)
```

---

## 성능 최적화 (Jetson Orin NX)

### 1. YOLO 모델 최적화
```bash
# 더 작은 입력 사이즈 사용 (속도 ↑, 정확도 ↓)
yolo_input_size:=416

# 더 가벼운 모델 사용
yolov8m.engine  # medium (yolov8l 대신)
yolov8s.engine  # small
```

### 2. 처리 속도 조절
```bash
# YOLO FPS 제한
max_fps:=8.0

# Fusion FPS 제한
max_fps:=8.0
```

### 3. 포인트클라우드 다운샘플링
```bash
# 더 큰 stride (속도 ↑, 정확도 ↓)
downsample_stride:=8
```

---

## 문제 해결

### 1. RTAB-Map 맵이 없다는 경고
```
No RTAB-Map cloud available yet
```
**해결**: RTAB-Map이 초기화될 때까지 기다리세요 (약 5-10초). 로봇을 천천히 움직이면서 VSLAM 초기화를 돕습니다.

### 2. TF transform 에러
```
TF transform map←camera_color_optical_frame unavailable
```
**해결**: RTAB-Map odometry가 정상 작동하는지 확인:
```bash
ros2 topic hz /rtabmap/odom
```

### 3. YOLO 모델 로드 실패
```
Failed to load model
```
**해결**:
- 모델 경로가 올바른지 확인
- TensorRT 버전이 호환되는지 확인 (Jetson용으로 재변환 필요할 수 있음)

### 4. CUDA Out of Memory
**해결**:
- 더 작은 YOLO 모델 사용 (yolov8s, yolov8m)
- 입력 크기 줄이기 (input_size:=416)
- 다른 GPU 프로세스 종료

---

## 성능 지표 (Jetson Orin NX 16GB 기준)

| 항목 | 값 |
|------|-----|
| YOLO 추론 속도 | ~10 FPS (YOLOv8l, 640x640) |
| 전체 파이프라인 | ~8 FPS |
| GPU 메모리 사용량 | ~3.5 GB |
| CPU 사용량 | ~40% (4코어) |
| 위치 정확도 | ±5 cm (RTAB-Map fusion 적용 시) |
| 크기 정확도 | ±10% (PCA 기반) |
| 방향 정확도 | ±15° (PCA yaw 추정) |

---

## 개발자 정보

- **패키지**: semantic_mapper_vslam
- **버전**: 1.0.0
- **ROS 2**: Humble
- **플랫폼**: Jetson Orin NX 16GB
- **카메라**: Intel RealSense D455
- **로봇**: Agile Tracer

---

## 라이선스

Apache 2.0

---

## 참고 자료

- [RTAB-Map ROS2](https://github.com/introlab/rtabmap_ros)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [RealSense ROS2](https://github.com/IntelRealSense/realsense-ros)
