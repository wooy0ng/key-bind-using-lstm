# 환경
- python 3.9.15
- librosa 0.10.0 0rc0

# code 실행
> python3 main.py

default mode = 'key_train'


# config.json 설정
- path:base = dataset_path 설정
- model:mode = default mode 설정
- model:num_epochs = classification model의 train에서 사용하는 epoch
- model:key_num_epochs = key generation model의 train에서 사용하는 epoch


# 주요 파일 설명
- main.py - 전체 코드 실행
- mode.py - train, test, key_train, key_test function 제공
- model.py - LSTMClassification / KeyGeneration model 제공 (model network 변경)


# Dataset 설명
- main 폴더에 있는 Dataset 폴더 (녹음했던 파일)
- train/test/validation: 각 mode 별로 true와 false 파일이 포함
- true/fasle: true 폴더에 있는 wav 파일이 true로 학습 (1) / false 폴더에 있는 wav 파일이 false로 학습 (0)


# MODE 별 코드 실행
### 1. train mode 실행 = train, test, key_train, key_test 실행
> python3 main.py --mode train

- Data Compression -> LSTM + Classification -> clf model 저장
- 저장된 clf model 로드 -> test 실행
- 저장된 clf model 호출 -> Key-generation -> key-gen model 저장
- 저장된 key-gen model 로드 -> key-generation test 실행

### 2. test mode 실행 = test 실행
> python3 main.py --mode test
- 저장된 clf model 로드 -> test 실행

### 3. key_train mode 실행 = key_train, key_test 실행
> python3 main.py --mode key_train
- 저장된 clf model 호출 -> Key-generation -> key-gen model 저장
- 저장된 key-gen model 로드 -> key-generation test 실행

### 4. key_test mode 실행 = key_test 실행
> python3 main.py --mode key_test
- 저장된 key-gen model 로드 -> key-generation test 실행