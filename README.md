# CCTV-Compression(YOLOv4 + Deepsort + Pruning)
- YOLOv4를 사용하여 객체 탐지(Bounding box, Classification)
- 탐지된 객체 정보를 통해 Deepsort를 사용하여 원하는 객체의 정지 상태나 움직임을 식별
- 식별된 움직임 정보를 통해 움직임이 없는 상태를 제거하여 저장 용량 축소
- ADMM 기반의 Weight Pruning 기법을 사용하여 객체 탐지 모델 경량화
<p align="center"><img src="./Tracker/img/small_restarea_reduction_prune640.gif"></p>
<br/>

## Workflow
<p align="center"><img src="./Tracker/img/Workflow.PNG"></p>

<br/>

## 시작하기

### 선행 설치 조건

- CUDA Toolkit version은 10.1을 권장

```
tensorflow-gpu==2.3.0rc0
opencv-python==4.1.1.26
lxml
tqdm
absl-py
matplotlib
easydict
pillow
```

### Conda
```
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### 학습된 YOLOv4 weights 가져오기(COCO로 학습)
- [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)에서 다운로드한 파일을 ./YOLOv4/data/ 경로에 추가
- 직접 학습시키고자 할 경우 'YOLOv4 훈련' 파트부터 실행

<br/>

## YOLOv4
### YOLOv4 훈련
- 훈련 파라미터는 ./YOLOv4/core/config.py에서 수정
- 데이터는 COCO dataset 기준이며 다른 데이터에 적용하기 위해서는 ./YOLOv4/data/dataset/val2017.txt 파일과 같이 수정 필요
- val2017.txt의 형태 : (이미지경로 x1,y1,x2,y2,class_label ...), class 정보는 ./YOLOv4/data/classes/ 경로에 위치(config.py에서 수정 필요)
```
python train.py
```

### YOLOv4 예측
#### 1) 학습된 YOLOv4 weights 사용하여 예측
```
python save_model.py --weight_origin True
python detect.py --weights ./checkpoints/yolov4-416 --image ./data/kite.jpg
```
#### 2) 새로운 데이터로 학습시킨 weights를 사용하여 예측
```
python save_model.py
python detect.py --weights ./checkpoints/yolov4-416 --image ./data/kite.jpg
```

<br/>

## ADMM 기반 Weight Pruning
- train 데이터는 YOLOv4와 동일
- 파라미터 수정은 yolo_admm_pruning.py 내에서 진행(epoch, k_step, all_percent(제거 퍼센트) 등)
- 각 알고리즘 스탭마다 최종 출력(가중치 제거된 weight 파일)은 ./YOLOv4/checkpoints/admm_pruning/ 폴더 내에 존재(weight_prune, after_admm, retraining)
- 예측은 YOLOv4 예측의 2) 파트로 진행

### Weight Pruning 훈련
```
python yolo_admm_pruning.py
```

<br/>

## Deepsort
- 객체 추적은 YOLOv4 훈련 weights 또는 Weight Pruning weights 모두 사용 가능
- 적용 데이터는 ./Tracker/data/video/ 경로에 위치
- 파라미터 수정은 object_tracker.py 내에서 진행(iou & score threshold 등)
- 최종 출력(움직임이 탐지된 이미지만 저장된 영상 & 탐지된 객체 이미지 및 정보)은 ./Tracker/outputs 폴더 내에 존재

#### 1) 학습된 YOLOv4 weights 사용하여 예측
```
python save_model.py --weight_origin True
python object_tracker.py --video ./data/video/cars.mp4 --output ./outputs/demo1.avi
```

#### 2) Weight Pruning weights 사용하여 예측
```
python save_model.py
python object_tracker.py --video ./data/video/cars.mp4 --output ./outputs/demo1.avi
```

<br/>

## 결과
### 기존 YOLOv4 weights 사용하여 추적(60초)

<p align="center"><img src="./Tracker/img/small_demo_not_reduction_origin2.gif"></p>

### Weight Pruning(제거 비율: 20%) weights 사용하여 추적(60초)

<p align="center"><img src="./Tracker/img/small_demo_not_reduction_prune2.gif"></p>

### 용량 축소 적용(45초)

<p align="center"><img src="./Tracker/img/small_demo_reduction_prune2.gif" width="640" height="320"></p>

<br/>

## 참조
```
Zhang, T., Ye, S., Zhang, K., Tang, J., Wen, W., Fardad, M., & Wang, Y. (2018). A systematic dnn weight pruning framework using alternating direction method of multipliers. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 184-199).

Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.

Wojke, N., Bewley, A., & Paulus, D. (2017, September). Simple online and realtime tracking with a deep association metric. In 2017 IEEE international conference on image processing (ICIP) (pp. 3645-3649). IEEE.
```
* [admm-pruning](https://github.com/KaiqiZhang/admm-pruning)
* [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
* [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)

<br/>

## 기여자

* 양동욱(dongwook412@naver.com)
* 황보성훈(thehb01@gmail.com)
