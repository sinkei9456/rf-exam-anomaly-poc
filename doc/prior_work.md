# 시험장 무선 부정 기기 탐지와 좌석 위치 추정: 선행 조사

- 작성: 2026-11-16
- 상태: 선행 조사만 함. 실험은 시작하지 않음(D4D ETRI 투고 뒤 검토 권함)
- 조사 범위: 웹 검색으로 찾은 논문·기사·제품·데이터셋. KCI·CNKI·IEEE Xplore 를 직접 검색하지는 않음

## 0. 한눈에 보기

시험장에서 부정 기기의 무선 신호를 **찾는 것**은 지금 기술로 충분히 할 수 있다. 이미 스마트 안경을 찾는 휴대폰 앱이 나와 있고, 국내에서는 한국방송통신전파진흥원(KCA)이 2026년 10월에 'AI 안경 전용 탐지 장비' 개발 용역을 발주했다. 그래서 "찾는다"만으로는 새 연구가 되기 어렵다.

**어느 좌석인지까지 짚는 것**은 사정이 다르다. 2012년 도쿄공대가 빈 강의실에서 LTE 휴대폰 위치를 좌석 간격보다 작은 오차로 찾은 연구가 있다. 하지만 BLE 이어폰이나 AI 안경처럼 약한 신호를, 사람이 앉아 있는 시험장에서, 좌석 단위로 찾은 공개 연구와 공개 데이터셋은 찾지 못했다. 남은 연구 틈은 이 부분이다. 현실적인 목표는 "정답 좌석 하나"보다 "의심 좌석을 2개에서 4개로 좁혀 감독관이 확인하게 하는 것"이다.

## 1. 읽기 전에: 용어

| 용어 | 뜻 |
|---|---|
| IQ 데이터 | SDR 수신기가 받은 전파를 그대로 숫자로 저장한 것. D4D 에서 HackRF 로 받은 .cs8 파일이 이것 |
| 스펙트로그램 | IQ 데이터를 가로축 시간, 세로축 주파수, 밝기 세기로 그린 그림. D4D 는 이 그림에서 YOLO 로 신호 사각형을 찾음 |
| 업링크 | 휴대폰이 기지국으로 보내는 신호. 휴대폰이 꺼져 있으면 시험장 안에서는 거의 없음 |
| BLE | Bluetooth Low Energy. 스마트 안경·스마트워치·센서가 쓰는 저전력 Bluetooth. 짧은 패킷을 띄엄띄엄 보냄 |
| Bluetooth Classic | 무선 이어폰이 음악·음성을 보낼 때 주로 쓰는 Bluetooth. BLE 보다 연속적으로 보냄 |
| RSSI | 받은 신호의 세기. 가까우면 크고 멀면 작다는 점을 이용해 거리를 어림함 |
| TDOA | 같은 신호가 여러 수신기에 도착한 시간 차이로 위치를 구하는 방법. 수신기 시간을 정확히 맞춰야 함 |
| AoA, CTE | AoA 는 신호가 들어온 방향으로 위치를 구하는 방법. BLE 5.1 에서는 송신기가 방향 측정용 신호(CTE)를 일부러 붙여 보내야 함. 부정 기기는 이 신호를 보내지 않음 |
| fingerprint 방식 | 좌석마다 미리 신호를 받아 두고, 나중에 받은 신호가 어느 좌석 기록과 가장 닮았는지로 위치를 정하는 방법 |
| top-k 정확도 | 모델이 고른 후보 k 개 안에 정답 좌석이 들어 있는 비율 |
| FPR | 부정 기기가 없는데 있다고 잘못 알리는 비율 |

## 2. 아이디어와 배경

### 2.1 계기

세무사 시험 AI 부정행위 시나리오 그림(안경 미착용판)에서 출발했다. 단추형 카메라가 시험지를 찍어 주머니 속 휴대폰으로 보내고, 휴대폰이 LTE/5G 로 외부 AI 서버에 보내 답을 받은 뒤, 개조한 계산기 화면에 답을 띄우는 구성이다. D4D 2026 드론 탐지(HackRF IQ 데이터 → 스펙트로그램 → YOLO)를 응용해 이런 기기의 신호를 찾고, 어느 좌석인지까지 좁히자는 것이 아이디어다.

### 2.2 시나리오 속 무선 구간

| 구간 | 연결 방식 | RF 로 잡히나 |
|---|---|---|
| 단추 카메라 → 휴대폰 | 유선 또는 Bluetooth | 유선이면 안 잡힘. Bluetooth 면 2.4 GHz 에서 잡힘 |
| 휴대폰 → AI 서버 | LTE/5G 업링크 | 잘 잡힘. 사진을 올릴 때 휴대폰이 세게 보냄 |
| 휴대폰 → 계산기 | BLE 또는 900 MHz 대 | 휴대폰이 보내는 신호는 잡힘. 계산기는 받기만 해서 신호를 내지 않음 |
| AI 안경을 쓰는 경우 | 안경과 휴대폰 사이 BLE·Wi-Fi | 2.4/5 GHz 에서 잡힘 |

### 2.3 시험장 전파 환경의 특징

- LTE 업링크 대역은 시험 중 정말 조용하다. 휴대폰을 끄거나 걷어 가고, 기지국 신호(다운링크)는 다른 주파수를 쓰기 때문이다. 이 대역은 AI 없이 세기 threshold 만으로도 잡힌다. 그래서 논문 심사에서 "왜 AI 가 필요한가"를 먼저 물을 것이다.
- 5G n78 은 기지국 신호와 휴대폰 신호가 같은 주파수를 나눠 쓰므로 조용하지 않다.
- 2.4 GHz 는 조용하지 않다. 학교 Wi-Fi 공유기, 감독관 기기, 옆 교실과 복도의 기기 신호가 섞인다. D4D 의 혼잡 배경 문제가 그대로 나타난다.
- 허용 기기가 있다. 연속혈당측정기·인슐린 펌프 같은 의료기기는 BLE 를 주기적으로 보낸다. 이것을 부정 기기로 잡으면 큰 문제가 된다.

## 3. 학술 연구

시험장 무선 부정 기기를 직접 다룬 학술 논문은 아래 네 편 정도다.

| 연구 | 대상 신호 | 한 일 | 결과 | 한계 |
|---|---|---|---|---|
| 도쿄공대 Sano 외 2012 (IEEE ICCS), Yu 외 2018 (IEICE Trans. Commun. E101-B(10)) | LTE 휴대폰 업링크 (1.9575 GHz, 폭 3.6 MHz) | 144석 강의실에서 휴대폰 위치를 좌석 단위로 추정 | 평균 오차가 좌석 간격 0.78 m 보다 작음. RSSI·TDOA 방식은 평균 오차 최대 7 m | 아래 3.1 참고 |
| 李竟铭 외 2018 (数据采集与处理 33(6), 1086–1093, 중국) | 초록에 없음 | 딥러닝으로 부정 신호를 판별하고 위치를 찾아 경보·휴대 단말 표시까지 하는 시스템 | 초록에 수치 없음 | 원문 확인 필요(대역·모델·정확도) |
| Barros 외 2021 (ICPEC, 포르투갈 비아나두카스텔루 공대) | Wi-Fi·Bluetooth | 교실용 SDR 탐지·차단 시스템 구조 제안. 탐지 때 위치도 추정한다고 적음 | 실험 수치 없음 | 구조 제안만 있음. 차단(잡음 송신)은 국내 전파법상 쓰기 어려움 |
| Lumos, Sharma 외 2022 (USENIX Security, 카네기멜런대) | Wi-Fi | 숙소에 숨긴 IoT 기기(몰래카메라 등)를 찾고 위치 추정 | 44개 기기, 6개 장소에서 식별 95 %, 위치 오차 중앙값 1.5 m | 사람이 휴대폰을 들고 30분 걸어 다녀야 함. Wi-Fi 패킷을 엿들어 씀 |

시험장 목적은 아니지만 방법이 같은 계열인 연구도 있다.

| 연구 | 한 일 | 결과 |
|---|---|---|
| WRIST, Nguyen 외 2021 (arXiv 2107.05114) | 100 MHz 폭 스펙트로그램에서 YOLO 계열로 Wi-Fi·Bluetooth·ZigBee 신호를 찾고 종류 구분 | 아주 혼잡한 실제 환경에서 mAP 90 |
| Lo 외 2020 (WACV) | 24 MHz 폭 스펙트로그램에서 신호를 사물처럼 찾음 | GPU 없는 노트북에서 실시간, mAP 81.82 |
| CamLoPA 2024 (arXiv 2409.15169) | 숨긴 무선 카메라의 위치를 신호 경로 분석으로 찾음 | 숙소 몰래카메라용 |
| Duke 대학 연구 (2026 보도) | 카메라 사진에서 스마트 안경 렌즈 속 도광판을 찾아 식별 | 약 80 %. 무선이 아닌 영상 방법 |

WRIST·Lo 외 연구가 D4D 와 같은 방식이다. 그래서 "스펙트로그램과 YOLO 로 BLE 를 찾는다"는 것 자체는 새롭지 않다.

### 3.1 도쿄공대 연구 자세히

가장 가까운 선행 연구라서 실험 조건을 따로 적는다.

- 수신기: 강의실 네 모서리에 안테나 1개짜리 4대, 가운데에 안테나 3개짜리 1대. 모두 RF 케이블로 중앙 처리 장치에 직접 연결
- 신호: LTE 휴대폰이 주기적으로 보내는 상향 기준 신호(SRS). 신호 모양을 미리 알고 있어서 수신 신호와 맞춰 보며 찾음
- 방법: 수신기 쌍마다 채널 응답의 상관을 구해 fingerprint 로 쓰고, 좌석마다 가우시안 분포로 학습한 뒤 가장 가능성 높은 좌석을 고름
- 측정: 좌석마다 휴대폰을 2 cm 씩 옮기며 두 번씩, 좌석당 20번 측정. 20개 중 1개를 빼서 평가하고 나머지 19개로 학습
- 결과: 평균 오차가 좌석 간격 0.78 m 보다 작음. 같은 데이터에서 RSSI·TDOA 는 최대 7 m

한계도 분명하다. 실험자 한 명만 있는 빈 방이었고, 학습과 평가가 같은 날 같은 빈 방에서 받은 측정이라 결과가 낙관적일 수 있다. 사람 30–40명이 앉아 몸으로 신호를 가리는 실제 시험 조건은 시험하지 않았다. LTE 신호 구조를 알아야 하는 방법이라 BLE 이어폰에 그대로 쓸 수도 없다.

## 4. 국내 운영·제도 현황 (2026년)

- **AI 안경 첫 적발**: 2026년 5월 토익 정기시험에서 2명이 적발됐다. 감독관이 시험 전에 눈으로 의심해 확인했고, 시험 무효에 4년 응시 제한을 받았다. 한국토익위원회는 감독관 교육을 강화하고 있다.
- **KCA 탐지 장비 용역**: 2026-10-26 발주, 기간 70일, 5천만 원. 감독관이 들고 다니는 소형 장비이고 완충 때 3시간 이상 작동해야 한다. Bluetooth 신호로 AI 안경인지와 제조사를 알아내고, Wi-Fi 통신 패턴으로 사진·영상 전송이나 AI 사용 가능성을 추정한다. 핵심 과제는 휴대폰·스마트워치·무선 이어폰과 구별해 오탐을 줄이는 것이다. 위치 추정 언급은 없다. 성능 검증은 2027년 초로 예상되고, KCA 스스로도 결과를 장담하기 어렵다고 했다. 대상은 국가기술자격(지난해 540종목, 응시 약 229만 명)과 산업인력공단의 국가전문자격(세무사, 공인노무사, 공인중개사 등)이다.
- **국정감사 지적**: 산업인력공단 모의 훈련에서 금속탐지기와 전파 탐지기가 AI 안경을 잡지 못했다.
- **수능 (2026-11-19)**: 시험장 1,312곳에 전담 인력을 1명씩 두고 금속탐지기를 1차 선별에 쓴다. 전파 탐지기는 전문 기관 시험을 거쳐 내년 수능 도입을 검토한다. AI 안경을 쓰면 그해 시험이 무효이고 다음 해 응시도 제한된다.
- **다른 시험**: 도로교통공단은 전파 탐지기를 갖고 있어 전국 학과시험장에 배치할 예정이다. 한국어능력시험(TOPIK)은 내년까지 해외 시험장으로 탐지기를 늘린다. 일부 교육청은 교원 임용시험에 몰래카메라·도청 탐지기를 샀지만 AI 안경을 잡는지는 확인되지 않았다.
- **민간 도구**: 42Maru 가 웹 브라우저로 주변 Bluetooth 기기를 훑는 ExamGuard 를 2026년 11월 공개했다.
- **전문가 의견**: KAIST 한준 교수는 시험장 주변 무선 기기가 많아 BLE·Wi-Fi 신호만으로 특정 스마트 안경을 구별하기 어렵고, 실제 시험장에서 충분히 검증해야 한다고 했다. KAIST 윤상호 교수는 전파 차단은 현실적으로 어렵고, 지금은 눈으로 확인하고 보안 스티커를 붙이는 방법이 가장 확실하다고 했다.
- **과거 사례**: 2005년 수능 때 국산·수입 전파 탐지기를 하나씩 시험했으나 탐지 능력이 떨어져 구입을 보류했다.

## 5. 해외 운영·제품·특허

- **스마트 안경 탐지 앱 (2026년 2월부터)**: iPhone 용 ZuckOff, Android 용 Nearby Glasses 등. BLE 광고 신호 속 제조사 번호로 Meta·Snap 안경을 찾는다. 안경이 휴대폰과 연결된 뒤에는 광고 신호를 거의 내지 않아 처음 켤 때 위주로 잡힌다. Meta Quest 같은 같은 회사 기기에서 오탐이 나고, 몸에 가리거나 방향이 바뀌면 RSSI 가 크게 흔들려 거리가 멀면 믿기 어렵다.
- **미국 SAT**: College Board 가 시험 중 스마트 안경을 금지했다.
- **중국**: 대학 입시와 공무원 시험에서 스마트 안경을 금지했다. 하루 6–12 달러에 AI 안경을 빌려 주는 사례가 보도됐다.
- **중국 가오카오 무선 감시**: 무선 감시 차량(2017 후난성)과 드론(2015 뤄양)을 쓴다. 시험 전에 전파 환경을 기록해 두고 시험 중 달라진 신호를 추적한다. 한 실제 사례에서는 부정 기기가 2초 송신하고 3분에서 5분씩 쉬어서 위치를 찾기 어려웠다고 한다.
- **휴대폰 탐지기**: Berkeley Varitronics 의 Wolfhound 같은 휴대형 제품. 셀룰러 대역을 보고, 방향 탐지 안테나를 들고 사람이 걸어 다니며 찾는다. 교도소·보안 시설용이다.
- **특허**:
  - CN108627876A (중국, 2018 출원): 좌석마다 소형 탐지기를 두고 유도식 무선 이어폰의 음성 대역 자기장을 잡아 LED 밝기로 좌석을 알려 준다. 2024년 철회 간주됐다. 이어폰 송신 스위치를 끄면 못 잡는다는 한계가 지적된다.
  - CN200989924Y (중국 실용신안): 시험용 무선 이어폰 탐지기
  - US 11374947: 광대역 스펙트로그램에서 딥러닝(Faster R-CNN)으로 Bluetooth·Zigbee 등 신호를 찾고 위치를 정하는 장치(시험장 전용은 아님)

## 6. 오픈 데이터셋

시험장, AI 안경, 이어폰 전용 공개 데이터셋은 찾지 못했다. 용도별로 쓸 만한 것을 정리한다.

### 6.1 탐지 학습용 (스펙트로그램 + 박스 레이블)

| 데이터셋 | 내용 | 쓸모와 한계 |
|---|---|---|
| **Fraunhofer, Wicht 외 2022** (MDPI Data 7(12):168, FORDATIS 저장소, CC BY 4.0) | 스펙트로그램 2만 장. Wi-Fi 프레임 약 36만 개, BLE 1 MHz·2 MHz 각 21,340개, Bluetooth Classic, 신호 겹침 77,600개에 박스 레이블. IQ 원본 포함. 실제 받은 신호를 채널 모델로 섞어 늘림 | **가장 쓸모 있음.** 이어폰이 쓰는 Bluetooth Classic 이 있고, D4D 파이프라인에 바로 넣을 수 있는 형태 |
| Aloui 외 2026 (Mendeley Data, CC BY 4.0) | USRP B210, 중심 2.438 GHz, 56 MS/s. 드론(DJI Phantom 4 Pro)·Wi-Fi·Bluetooth 스펙트로그램 6,842장, 사각형 22,375개, YOLO 형식 | D4D 와 같은 형식. IQ 원본은 없음 |
| WRIST, Nguyen 외 2021 | 2.4 GHz 에서 Wi-Fi·Bluetooth·ZigBee 등 레이블 약 100만 개 | 공개 범위와 IQ 포함 여부 확인 필요 |

### 6.2 기기 구분용 (IQ 데이터)

| 데이터셋 | 내용 | 쓸모와 한계 |
|---|---|---|
| SDR4IoT (Zenodo 4639390, CC BY 4.0) | USRP N210 으로 받은 BLE·Zigbee IQ, 약 79 MB | 바로 내려받을 수 있지만 작음 |
| 오리건주립대 Albousayri·Hamdaoui 2026 (arXiv 2611.00332) | BLE 송신기 31대, USRP B210 2대, 6 MS/s, NumPy complex64 | 31대 모두 ESP32-C3 개발 보드라 실제 이어폰·안경이 아님 |
| Jagannath 외 2023 (arXiv 2303.13538) | 노트북·어댑터·라즈베리 파이 등의 상용 Wi-Fi·Bluetooth 칩, 약 72 GB | 내려받는 경로와 라이선스 확인 필요 |
| Wearable Device Bluetooth/BLE Physical Layer Dataset (MDPI Data 9(4):53, 2024) | 이름은 웨어러블 기기 Bluetooth 데이터 | 페이지 접근이 막혀 내용 미확인 |

### 6.3 LTE 업링크

| 데이터셋 | 내용 | 쓸모와 한계 |
|---|---|---|
| NIST 업링크 IQ (AWS-3 대역) | 실제 LTE 업링크 IQ, 61.44 MS/s, 스펙트로그램 그림 포함 | 휴대폰 송신 신호 모양 참고용 |
| WASD (IEEE DataPort, 무료 계정) | 실제 LTE·5G 19개 대역 배경에 합성 이상 신호를 넣고 박스 레이블(CSV) | D4D 와 같은 "합성 신호 + 실제 배경" 발상 |

### 6.4 위치 추정

| 데이터셋 | 내용 | 쓸모와 한계 |
|---|---|---|
| UVIndoorLoc-BLE-AoA&IQ (Zenodo 19823027, 발렌시아대 2026, CC BY 4.0) | 수신기 4대(안테나 8개짜리), 실험실·복도·회의실·강당, IQ·RSSI·방향, 약 1.6 GB | 태그가 방향 측정용 신호(CTE)를 일부러 보내는 협조형. 부정 기기는 이 신호를 안 보내서 그대로 쓸 수 없음. 방법 개발 참고용 |
| Girolami 외 2023 Bluetooth 5.1 데이터셋 | 수신기 4대, 태그 4개, 정지·이동 상황 | 같은 이유로 참고용 |
| Furfari 외 2026 (Zenodo 18833426) | 수신기 9대(벽·천장), 웨어러블 태그 2개, 110 m² | 같은 이유로 참고용. IQ 포함 여부 확인 필요 |

### 6.5 가진 데이터

- D4D 실내 드론 없음 세션(contest/d4d_2026 의 data/raw/hackrf/1_nodrone_indoor): HackRF, 2450 MHz, 20 MSps, 1초 세션 5개씩, 2026-10-06·07 취득. 교실은 아니지만 2.4 GHz 실내 배경 참고로 쓸 수 있다.

### 6.6 없는 데이터

**협조하지 않는 기기(이어폰·안경) + 사람이 앉은 교실 + 좌석 정답**이 함께 있는 데이터는 없다. 직접 취득해야 하고, 이 데이터를 공개하는 것 자체가 논문 기여가 된다.

## 7. 현실성 판단

| 단계 | 현실성 | 이유 |
|---|---|---|
| 신호 발견 | 높음 | 앱과 탐지기가 이미 함. 시험 중엔 휴대폰을 꺼서 업링크 대역이 조용함 |
| 기기 종류 구분 (안경, 이어폰, 허용 의료기기) | 중간 | 제조사 번호가 보이면 쉬움. 안 보이면(연결 중, 저가 무명 제품, 개조품) 같은 BLE 칩끼리는 스펙트로그램 모양이 거의 같아서 송신 주기와 전송량 패턴으로 가려야 함 |
| 좌석 단위 위치 | 낮음에서 중간 | RSSI 만 쓰면 교실에서 오차가 수 m. 좌석 단위가 나오려면 수신기 여러 대를 시간 맞춰 묶어야 하는데 2.4 GHz 에서 그렇게 묶는 저가 장비가 드묾. BLE 이어폰은 휴대폰 LTE 보다 송신 세기가 수십 배 약하고 몸에 가리면 크게 흔들려, 빈 방에서 만든 학습 데이터가 사람이 앉으면 맞지 않을 가능성이 큼 |
| 부정행위 막기 | 부분적 | 송신 스위치를 끄거나, 카메라를 유선으로 잇고 AI 를 오프라인으로 돌리면 RF 흔적이 없음. 금속탐지기·눈 확인과 같이 쓰는 보조 수단 |

운영에서 주의할 점도 있다. 오탐은 무고한 수험생을 의심하는 결과로 이어지므로 FPR 을 아주 낮게 잡아야 하고, 최종 판단은 감독관이 해야 한다. 신호 내용은 해독하지 않고 세기와 모양만 보는 수동 수신으로 해야 통신비밀 문제를 피할 수 있다. 전파 차단은 국내에서 쓰기 어려우므로 탐지와 위치 추정 쪽 수요가 있다.

종합하면, 교실에 수신기 여러 대를 고정 설치하고 의심 좌석 후보를 좁혀 사람이 앉은 실제 교실에서 평가하는 연구는 찾은 범위에서는 없다. KCA 용역은 휴대형 근접 탐지라 겹치지 않는다. 목표는 정답 좌석 하나가 아니라 top-3 같은 좌석 후보로 잡는 것이 현실적이다.

## 8. 진행한다면

### 8.1 D4D 에서 가져올 것과 새로 할 것

| 구분 | 내용 |
|---|---|
| 가져올 것 | HackRF·라즈베리 파이 패키지, 스펙트로그램 → YOLO 파이프라인, "합성 신호 + 실제 배경 IQ 데이터" 학습 방법, 처음 보는 장소의 FPR 평가 틀(처음 보는 교실로 바꿈) |
| 다시 정할 것 | 스펙트로그램 해상도. BLE 패킷은 수백 µs 길이에 폭 1–2 MHz 라 드론 영상 신호보다 훨씬 짧고 좁음 |
| 다시 정할 것 | 감시 대역. HackRF 는 한 번에 20 MHz 만 보는데 2.4 GHz 전체(약 83 MHz), LTE 업링크 여러 대역, 900 MHz 대를 봐야 함. 넓은 대역의 세기만 빠르게 훑는 hackrf_sweep 으로 의심 대역을 찾고 그 대역만 IQ 로 받는 2단계 구성 검토 |
| 새로 할 것 | 위치 추정(D4D 에 없음), 수신기 여러 대 구성, 사람이 앉은 교실 데이터 취득(학교·시험장 협조가 가장 큰 장애) |

### 8.2 첫 판단 실험 (진행 여부 결정용)

1. 저가 수신기 4대에서 8대를 빈 교실의 모서리와 줄마다 둔다.
2. BLE 이어폰을 좌석마다 옮기며 세션을 받는다.
3. 사람들이 앉은 상태에서 같은 측정을 반복한다.
4. 두 조건의 위치 오차와 top-3 정확도를 비교한다.

사람이 앉아도 정답 좌석이 대부분 top-3 안에 들어오면 진행하고, 아니면 접는다. HackRF 1대로는 위치를 잴 수 없어 수신기를 새로 사야 하므로 장비 구입은 승인 뒤에 한다.

### 8.3 순서

D4D ETRI 투고를 먼저 끝내고 시작하는 것을 권한다. 시작하기 전에 9장의 원문 확인을 먼저 한다.

## 9. 남은 확인 사항

- 李竟铭 외 2018 원문: 대역, 모델, 위치 정확도, 좌석 단위인지 시험장 단위인지
- Sano 외 2012 원문: 사람이 있는 조건을 따로 시험했는지
- WRIST 데이터 공개 범위, Jagannath 외 72 GB 데이터의 내려받는 경로, MDPI 2024 웨어러블 데이터셋 내용
- KCA 용역 결과(2027년 초 검증 예정)
- KCI·CNKI·IEEE Xplore 직접 검색(이번 조사는 웹 검색 기준)

## 10. 출처

학술 연구
- [Yu 외 2018, Fingerprint-based radio emitter localization 안내 (arXiv 1804.02124, 도쿄공대 시험장 LTE 위치 추정 포함)](https://arxiv.org/abs/1804.02124)
- [李竟铭 외 2018, 基于深度学习的无线电作弊信号发现与定位系统](https://sjcj.nuaa.edu.cn/sjcjycl/article/abstract/20180617?st=search)
- [Barros 외 2021, A System Architecture to Detect and Block Unwanted Wireless Signals in a Classroom](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.ICPEC.2021.12)
- [Lumos, USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/sharma-rahul)
- [WRIST, Spectro-Temporal RF Identification using Deep Learning](https://arxiv.org/pdf/2107.05114.pdf)
- [Lo 외 2020, IoT Discovery Using Deep Neural Networks (WACV)](https://openaccess.thecvf.com/content_WACV_2020/papers/Lo_Internet_of_Things_IoT_Discovery_Using_Deep_Neural_Networks_WACV_2020_paper.pdf)
- [CamLoPA, arXiv 2409.15169](https://arxiv.org/pdf/2409.15169)

국내 현황
- [SBS Biz: 국가 시험장 AI 컨닝 전용 탐지기 개발 착수](https://biz.sbs.co.kr/amp/article/20000343272)
- [시대일보 2026-11-05: 시험별 AI 안경 대응](https://www.sidae.com/article/2026110512523337520)
- [Korea JoongAng Daily: 수능 앞 AI 안경 대책](https://www.koreajoongangdaily.com/korea/hidden-in-plain-sight-schools-hunt-for-ai-smart-glasses-ahead-of-college-entrance-exam/12965932)
- [한국경제: 토익 AI 글라스 커닝 첫 적발](https://www.hankyung.com/article/202606107661Y)
- [SBS Biz: AI 안경 초비상](https://biz.sbs.co.kr/amp/article/20000315816)
- [파이낸셜뉴스: 국가기술자격 스마트글라스 금지 추진](https://www.fnnews.com/news/202606100728543326)
- [한교닷컴: 수능 부정 방지 장비 미비(2005 전파탐지기 보류)](https://hangyo.com/mobile/article.html?no=14009)

해외 현황·제품·특허
- [Help Net Security: Nearby Glasses 앱](https://www.helpnetsecurity.com/2026/02/27/nearby-glasses-android-app-detect-smart-glasses/)
- [MobileSyrup: 스마트 안경 탐지 앱](https://mobilesyrup.com/2026/02/25/someone-made-an-app-to-detect-nearby-smart-glasses/)
- [Cybernews: 시험에서 스마트 안경 금지](https://cybernews.com/tech/smart-glasses-banned-cheating-exams/)
- [Gadget Review: 스마트 안경 시험 부정](https://www.gadgetreview.com/smart-glasses-turn-exams-into-open-book-tests-schools-cant-stop-it)
- [POLICE: Wolfhound 휴대폰 탐지기](https://www.policemag.com/news/berkeley-varitronics-introduces-wolfhound-cellular-detector)
- [후난성: 무선 감시 차량으로 가오카오 부정 신호 포착](http://gxt.hunan.gov.cn/wxd_ycl/gldt_ycl/w_szcz_ycl/201706/t20170606_8128615.html)
- [CSMonitor: 중국 드론으로 시험 부정 단속](https://www.csmonitor.com/World/Global-News/2015/0608/Why-is-China-using-drones-to-catch-college-exam-cheaters)
- [CN108627876A](https://patents.google.com/patent/CN108627876A/zh)
- [CN200989924Y](https://patents.google.com/patent/CN200989924Y/en)

데이터셋
- [Fraunhofer, Spectrogram Data Set for Deep-Learning-Based RF Frame Detection](https://mdpi.com/2306-5729/7/12/168)
- [Aloui 외 2026, Dataset_Drones_Bluetooth_Wifi (Mendeley)](https://data.mendeley.com/datasets/y4w96kzsb5/1)
- [SDR4IoT BLE & Zigbee RF dataset (Zenodo)](https://zenodo.org/records/4639390)
- [Albousayri·Hamdaoui 2026, Bluetooth Device Fingerprinting dataset](https://arxiv.org/html/2611.00332)
- [Jagannath 외 2023, Bluetooth and WiFi Dataset for Real World RF Fingerprinting](https://arxiv.org/abs/2303.13538)
- [Wearable Device Bluetooth/BLE Physical Layer Dataset](https://doi.org/10.3390/data9040053)
- [NIST 업링크 IQ](https://christopher-catalog-dev.app.cloud.gov/dataset/uplink-iq-recordings)
- [WASD, Wireless Anomaly Signal Dataset](https://ieee-dataport.org/open-access/wasd-wireless-anomaly-signal-dataset)
- [UVIndoorLoc-BLE-AoA&IQ (Zenodo)](https://zenodo.org/records/19823027)
- [Girolami 외 2023, Bluetooth 5.1 AoA·RSS 데이터셋](https://www.researchgate.net/publication/372869182_A_Bluetooth_51_Dataset_Based_on_Angle_of_Arrival_and_RSS_for_Indoor_Localization)
- [Furfari 외 2026, AoA·RSS 실내 위치 데이터셋 (Zenodo)](https://zenodo.org/records/18833426)
