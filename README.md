# 🔍 ReCheck (리체크)

> **AI 기반 중고 자산 관리 및 다차원 사기 방어 시스템**
> 
> 중고 거래 시장의 정보 비대칭 문제를 해결하고, AI 분석을 통해 안전하고 투명한 거래 생태계를 구축합니다.

<br>

## 🛠 **Tech Stack**

| Category | Technologies |
| :--- | :--- |
| **AI / ML** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white) `Multi-modal Analysis` `Anomaly Detection` |
| **Backend** | ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi) ![Spring Boot](https://img.shields.io/badge/Spring_Boot-6DB33F?style=flat-square&logo=spring-boot&logoColor=white) |
| **Frontend** | ![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=black) ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) |
| **Database** | ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=PostgreSQL&logoColor=white) ![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=Redis&logoColor=white) |

<br>

## ✨ **Key Features**

### 1️⃣ **Smart Item Scanner (Seller)**
* **AI Model Recognition**: 이미지 업로드 시 브랜드 및 상세 모델명 자동 식별
* **Real-time Pricing**: 실시간 시장 데이터 기반 최적 판매가(Resell Price) 제안
* **Condition Grading**: 픽셀 단위 분석을 통한 객관적 상태 등급(S~C) 부여

### 2️⃣ **Fraud Detection Signal (Buyer)**
* **URL Risk Analysis**: 게시글 링크 입력 시 AI가 위험도를 3단계(안전/주의/위험)로 시각화
* **Seller Reliability**: 과거 거래 이력 및 활동 패턴 기반의 통합 신뢰도 점수 제공

### 3️⃣ **Market Insight & Prediction**
* **Price Trend Analysis**: 시계열 분석을 통한 향후 시세 흐름 예측
* **Trading Timing**: 계절성 및 트렌드 분석을 통한 최적의 거래 시점 추천

<br>

## 🧬 **Core Technology**

* **Multi-modal Risk Scanning**: 이미지, 텍스트, 시세, 평판 데이터를 통합 분석하는 앙상블 모델 적용
* **Anomaly Detection**: 로고 폰트, 자간, 봉제 패턴의 미세 오차를 감지하여 가품 여부 판단 보조

<br>

## 🚀 **Git Workflow Guide**

본 프로젝트는 효율적인 협업을 위해 아래와 같은 Git 전략을 준수합니다.

### 1. Issue & Branch 전략
* 새로운 작업 시작 전 **GitHub Issue** 생성 (예: `#22`)
* 브랜치 네이밍: `feat/issue-<number>` 또는 `fix/issue-<number>`
* 예시: `git checkout -b feat/issue-22`

### 2. Commit Message Convention
* `feat`: 새로운 기능 추가
* `fix`: 버그 수정
* `docs`: 문서 수정 (Readme 등)
* `refactor`: 코드 리팩토링
* **Format**: `type: description #issue-number` (예: `feat: add AI detection logic #22`)

### 3. Pull Request & Merge
* 작업 완료 후 `main` 브랜치로 PR 생성
* 팀원의 리뷰 및 승인 후 Merge 진행

<br>

## 👥 **Team Members**

* **이채은 ([@chaeeun7201](https://github.com/chaeeun7201))** - AI Modeling & Project Lead
* **Backend Lead** - System Architecture & API Design
* **Frontend Lead** - UI/UX Research & Interaction Design

---
© 2026 ReCheck Team. All rights reserved.
