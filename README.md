# ForensicPhotoSearch

이미지 1장을 Query로 입력하여 이미지 및 영상 데이터에서
동일인 또는 유사 인물을 검색하고,
영상 내 등장 시간과 증거 정보를 함께 제공하는
포렌식 인물 검색 도구입니다.

---

## Overview

본 프로젝트는 디지털 포렌식 환경에서 확보된 이미지 및 영상 데이터에 대해
Query 이미지에 나타난 인물을 기준으로 관련 증거를 검색하는 것을 목표로 합니다.

현재 시스템은 다음과 같은 검색 파이프라인을 구성하고 있습니다.

```text
Query Image
     ↓
Person Detection
     ↓
Person Re-ID Feature Extraction
     ↓
Image Search / Video Search
     ↓
Video Tracking
     ↓
Similarity Calculation
     ↓
Track-level Scoring
     ↓
Match Segment Extraction
     ↓
Evidence Generation
     ↓
Integrated Search Result
Main Features
1. Image Person Search

Query 이미지의 인물 특징을 추출한 뒤,
검색 대상 이미지에서 사람을 검출하고 인물 영역을 추출하여
Query와의 유사도를 계산합니다.

현재 이미지 검색에서는 YOLO 기반 Person Detection과
OSNet 기반 Person Re-ID를 사용합니다.

2. Video Person Search

영상에서는 사람을 검출하고 Tracking하여
각 인물에게 Track ID를 부여합니다.

각 Track에 대해 일정 간격으로 Re-ID를 수행하고
Query 이미지와의 Cosine Similarity를 계산합니다.

Video
 ↓
YOLO Person Detection
 ↓
BoT-SORT Tracking
 ↓
Track ID
 ↓
Person Crop
 ↓
OSNet Re-ID
 ↓
Cosine Similarity
3. Track-level Scoring

단일 프레임의 유사도만 사용하는 것이 아니라
Track 전체의 유사도를 기반으로 추가적인 점수를 계산합니다.

현재 사용하는 지표:

Best Score
Average Score
High Score Count
High Score Ratio
Track Score

현재 Baseline Track Score:

Track Score =
    0.5 × Best Score
  + 0.3 × Average Score
  + 0.2 × High Score Ratio

※ 현재 가중치는 초기 실험을 위한 Baseline이며,
향후 실험 데이터를 기반으로 조정할 예정입니다.

4. Match Segment Extraction

영상 내에서 Query와 높은 유사도를 보이는 구간을 추출합니다.

현재 High Score Threshold:

0.75

예:

Track 13


00:09.000 ~ 00:14.333
00:15.000 ~ 00:15.167
00:15.500 ~ 00:16.333

각 Match Segment에는 다음 정보가 포함됩니다.

Start Frame
End Frame
Start Time
End Time
Duration
Best Score
Best Frame
Evidence
5. Evidence Generation

검색 결과에 대한 증거 이미지를 자동으로 생성합니다.

Track 단위:

before.jpg
best_frame.jpg
after.jpg

Match Segment 단위:

segment_01/
├── before.jpg
├── best.jpg
└── after.jpg

이를 통해 단일 최고점 프레임뿐만 아니라
전후 상황을 함께 확인할 수 있습니다.

6. Multi-Video Search

하나의 Query 이미지를 사용하여
여러 영상을 한 번에 검색할 수 있습니다.

예:

video/data/
├── query.jpg
├── test.mp4
└── test2.mp4

각 영상에 대해 독립적으로 Tracking 및 Re-ID를 수행한 뒤
전체 결과를 하나의 Ranking으로 통합합니다.

예:

test.mp4
 ├── Track 13
 ├── Track 1
 └── Track 26


test2.mp4
 ├── Track 101
 ├── Track 76
 └── Track 89

영상별 Evidence도 분리하여 관리합니다.

video/evidence/
├── test/
│   ├── track_1/
│   └── track_13/
│
└── test2/
    ├── track_76/
    └── track_101/
Result Structure

검색 결과에는 다음과 같은 정보가 포함됩니다.

Result
├── source_type
├── result_id
├── similarity
├── best_score
├── average_score
├── high_score_count
├── sample_count
├── high_score_ratio
├── track_score
├── video
├── track_id
├── best_frame
├── best_time
├── first_time
├── last_time
├── duration
├── match_segments
└── evidence

검색 결과는 다음 파일에 저장됩니다.

results/search_results.json
Evidence Structure

현재 영상 Evidence 구조:

video/
└── evidence/
    ├── test/
    │   ├── track_1/
    │   ├── track_13/
    │   └── ...
    │
    └── test2/
        ├── track_76/
        ├── track_89/
        └── ...

각 Track에는 분석 과정에서 생성된 프레임과
Evidence 이미지가 저장됩니다.

Technologies
Detection / Tracking
Ultralytics YOLO11n
BoT-SORT
Person Re-Identification
Torchreid / deep-person-reid
OSNet
Deep Learning
PyTorch
Torchvision
Image / Video Processing
OpenCV
Language
Python
Project Structure
ForensicPhotoSearch/
│
├── forensic_search.py
├── reid_model.py
├── similarity.py
├── person_detector.py
├── person_cropper.py
├── image_loader.py
├── image_database.py
├── build_embeddings.py
│
├── video/
│   ├── video_search.py
│   ├── tracker.py
│   ├── evidence_manager.py
│   └── ...
│
├── deep-person-reid/
│
├── data/
├── results/
└── video/
Installation
git clone https://github.com/Karsel83/ForensicPhotoSearch.git


cd ForensicPhotoSearch


py -3.13 -m venv .venv


.\.venv\Scripts\Activate.ps1


python -m pip install --upgrade pip


python -m pip install -r requirements.txt

PowerShell 실행 정책 오류가 발생하는 경우:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned


.\.venv\Scripts\Activate.ps1
Usage

Query 이미지와 검색 대상 영상을 준비합니다.

video/data/
├── query.jpg
├── test.mp4
└── test2.mp4

프로젝트 루트에서 실행:

python forensic_search.py

검색 결과는 다음 위치에 저장됩니다.

results/search_results.json
Example Result
======================================================================
INTEGRATED SEARCH RESULTS
======================================================================


[Rank 1] VIDEO
  Similarity : 0.8983
  TrackScore : 0.7520
  Avg Score  : 0.7156
  High Ratio : 0.4409
  Video      : test.mp4
  Track      : 13
  Time       : 00:09.333
  Duration   : 15.633s


  Match Segments:
    00:09.000 ~ 00:14.333 (best=0.8983)
    00:15.000 ~ 00:15.167 (best=0.7917)
    00:15.500 ~ 00:16.333 (best=0.8040)
Current Status
Completed
 Image Person Search
 Video Person Detection
 Multi-object Tracking
 Person Re-ID
 Image / Video Unified Search
 Track-level Scoring
 Match Segment Extraction
 Track Evidence Generation
 Segment Evidence Generation
 Multi-Video Search
 Video-specific Evidence Separation
 JSON Search Result Generation
 Video-level Result Separation
In Progress
 SHA-256 / MD5 / pHash
 Exact Duplicate Detection
 Visual Duplicate Detection
 Original Evidence / AI Analysis Separation
 Evidence Integrity Manifest
 Multi-Model Evaluation
 Clustering-based Evaluation
 Automated Experiment Framework
 GUI
 Quantitative Performance Evaluation
Planned Research

향후 여러 Person Re-ID 모델을 동일한 데이터셋에서 비교하고,
각 모델의 Embedding에 대해 다양한 Clustering 알고리즘을 적용하여
성능을 평가할 예정입니다.

예정된 실험 구조:

Re-ID Models
├── Model A
├── Model B
└── Model C


        ×


Clustering Algorithms
├── K-Means
├── MiniBatch K-Means
├── DBSCAN
├── HDBSCAN
├── Agglomerative Clustering
├── Gaussian Mixture Model
├── Spectral Clustering
├── BIRCH
└── OPTICS

이를 통해 Re-ID 모델과 Clustering 방법의 조합에 따른
인물 검색 및 군집화 성능을 비교할 예정입니다.

예정 평가 지표:

Retrieval
Rank-1
Rank-5
Rank-10
mAP
Clustering
ARI
NMI
Purity
Silhouette Score
Calinski-Harabasz Index
Davies-Bouldin Index
Research Direction

최종적으로 다음과 같은 통합 실험 환경을 구축하는 것을 목표로 합니다.

Common Dataset
      ↓
Common Detection / Tracking
      ↓
┌──────────────┬──────────────┬──────────────┐
│   Model A    │   Model B    │   Model C    │
└──────────────┴──────────────┴──────────────┘
      ↓              ↓              ↓
  Embedding      Embedding      Embedding
      ↓              ↓              ↓
      └──────────────┼──────────────┘
                     ↓
          Clustering Algorithms
                     ↓
             Evaluation
                     ↓
       CSV / JSON / Visualization

동일한 데이터셋과 동일한 전처리 환경에서
모델별 성능을 비교하는 것을 목표로 합니다.

Open Source / Attribution

본 프로젝트는 다음 오픈소스 프로젝트 및 사전학습 모델을 활용합니다.

Ultralytics YOLO
Torchreid / deep-person-reid
OSNet
PyTorch
Torchvision
OpenCV

각 프로젝트의 라이선스 및 사용 조건을 확인하여 사용합니다.

Disclaimer

본 프로젝트는 연구 및 교육 목적의 프로토타입입니다.

검색 유사도 점수만으로 동일인을 확정하지 않으며,
실제 수사 및 법적 판단에는 추가적인 검증과 전문가의 판단이 필요합니다.