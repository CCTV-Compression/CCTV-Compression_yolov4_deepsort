# CCTV-Compression_yolov4_deepsort
- yolov4를 사용하여 객체 탐지(Bounding box, Classification)
- 탐지된 객체 정보를 통해 deepsort를 사용하여 원하는 객체의 정지 상태나 움직임을 식별
- 식별된 움직임 정보를 통해 움직임이 없는 상태를 제거하여 저장 용량 축소
- ADMM 기반의 Weight Pruning 기법을 사용하여 객체 탐지 모델의 경량화
<br/>
<p align="center"><img src="./Tracker/img/Workflow.PNG"></p>

## 시작하기

### 선행 설치 조건

CUDA Toolkit version은 10.1을 권장

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

### 데이터

(프로젝트에 필요한 전체 데이터는 보안상 업로드가 불가하여 직접 크롤링 및 라벨링한 데이터(damage1100.csv)만 업로드)

문장을 이진 분류(0/1) 레이블링 함.
||문장|T/F|
|------|---|---|
|1|산사태와 축대 붕괴 침수 피해 등에 철저히 대비해 주시기 바랍니다.|0|
|2|천만다행으로 지나던 사람이 없어 인명피해는 없었습니다.|0|
|3|최근 찜통 더위가 연일 이어진 가운데 전 남지역에서 가축과 양식 어류 폐사가 속출하고 있습니다.|1|
|4|중국 남동부 일대에 연일 불볕 더위가 이어지면서 열사병으로 20여 명이 숨진 것으로 파악됐습니다.|1|

## 파일 설명
#### 1) preprocessing.ipynb \& preprocessing_Ensemble.ipynb

    빈 텍스트 제거, Stop word 제거, 품사 태깅

#### 2) Sentence_classification.ipynb

    벡터화(Word2Vec, Doc2Vec, Fasttext) 및 분류 모델(LSTM, 1D-CNN, XGBoost) 생성

#### 3) sentence_utility.py

    1), 2)에 필요한 함수 구현

#### 4) Associative_classification(TBM).ipynb

    연관 분류 모델('Associative classification based on the Transferable Belief Model') 생성 

#### 5) Predict.ipynb

    생성된 모델들을 통해 적용 및 성능 평가

## 기여자

* [양동욱](dongwook412@naver.com)
* [황보성훈](thehb01@gmail.com)
