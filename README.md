# ReCheck
## AI 기반 중고 자산 관리 및 사기 방어 시스템 

-----
📌 Project Overview

중고 거래 시장에서 발생하는 가품 사기 및 정보 비대칭 문제를 해결하기 위한 AI 솔루션입니다. 판매자에게는 합리적인 판매가를 제안하고, 구매자에게는 다각도의 데이터 분석을 통해 거래 안전도를 제공합니다. 

✨ Key Features


1️⃣ [판매자] 스마트 물품 스캐너 

AI 모델 인식: 사진 업로드 시 브랜드와 모델명을 자동으로 식별합니다. 

실시간 리셀가 산출: 현재 시장 데이터를 분석해 적정 판매 가격을 제안합니다. 

상태 등급 평가: 스크래치나 오염을 정밀 분석해 객관적인 등급(S~C)을 부여합니다. 


2️⃣ [구매자] 사기 탐지 신호등 

URL 리스크 분석: 게시글 링크 입력 시 AI가 위험도를 평가해 3단계 신호등(안전/주의/위험)으로 표시합니다. 

판매자 신뢰도 검증: 과거 거래 이력과 활동 패턴을 종합 분석합니다. 


3️⃣ 시세 예측 차트

시계열 분석: 과거 가격 추이 데이터를 분석해 향후 6개월~1년의 시세를 예측합니다.

최적 타이밍 제안: 계절성과 시장 트렌드를 반영해 가장 유리한 거래 시점을 알려줍니다. 


🛠 Tech Stack (Proposed)

AI/ML: Python, PyTorch (Multi-modal Risk Scanning, Anomaly Detection) 

Backend: Java Spring Boot (Main Server), Python FastAPI (AI Server) 

Frontend: HTML/CSS/JS or React 

Database: PostgreSQL, Redis 


🧬 Core Technology

Multi-modal Risk Scanning: 이미지, 텍스트, 시세, 평판 4가지 데이터를 통합 분석합니다. 

Anomaly Detection: 픽셀 단위의 로고 형태, 폰트 자간, 봉제 패턴 미세 오차를 감지합니다.


-----
👥 Team

팀장: 이채은 (AI 모델링 및 프로젝트 총괄) 

팀원: 백엔드(Spring Boot), 프론트엔드 담당자
