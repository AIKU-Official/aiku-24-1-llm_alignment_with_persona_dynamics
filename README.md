# 👮 LAPD: Large language model Alignment through Persona Dynamics

## AIKU 24-Spring Project

**Authors**: 박서현(팀장), 성준영, 이종훈, 황정현

AIKU 2024년 전반기 프로젝트 / 고려대학교 CS471n 자연어처리 4팀

## 프로젝트 개요
본 프로젝트에서는 LLM이 특정 페르소나(MBTI)로 파인 튜닝되었을 때의 모델의 행동 및 선호의 양상을 관찰하였습니다. 모델은 LLaMA-3 Instruct 8B를, DPO를 사용하여 학습되었으며, 학습 데이터는 각 MBTI 별 특징을 반영하도록 GPT-4를 활용하여 제작되었습니다. 
학습 후 모델들을 AI Alignment Benchmark들을 통해 모델의 변화를 정량, 정성적으로 평가하였습니다. 
각 모델별로 MBTI의 성향이 발현된 결과들을 포착하였습니다. 특히, 새로 학습한 MBTI 특징들이 발현되어 약물, 범죄 등과 관련된 비윤리적 질의에 효과적으로 대응하고, 시련에 처한 사람의 상황에 더 깊게 공감하는 등 향상된 도덕성 및 사회성의 가능성을 보여주었습니다. 
학습에 활용된 데이터셋은 본 레포지토리의 data 폴더에 위치하며, 아래의 과정을 통해 생성한 데이터로 모델을 학습시킬 수 있습니다.
더 많은 정보를 얻고싶으시다면, [AIKU 노션](https://www.notion.so/aiku/Mamihlapinatap-ai-d0100f9c85424342bd47a2c496ebe25e)에 방문하셔서 확인해 주세요!

## 환경 구축

* Python 3.10+
* Linux
* NVIDIA GPU + CUDA CuDNN
* CUDAToolKit >= 11.8
* Pytorch >= 2.2
* Unsloth 가상환경

Anaconda를 사용하여 본 레포지토리를 사용하기를 권장합니다. 

환경 구축 방법은 다음과 같습니다:

1. **Unsloth 가상환경 설치**:

CUDA 버전에 맞추어 pytorch-cuda 버전을 설정하시길 바랍니다.
CUDA 12.1을 사용하실 경우엔, pytorch-cuda=12.1로 지정하면 됩니다. 
```
conda create --name unsloth_env \
    python=3.10 \
    pytorch-cuda=<your_CUDA_version> \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

```

2. **레포지토리 복제 및 가상환경 활성화**:
   본 레포지토리를 복제하고, 1에서 구축한 unsloth_env 가상환경을 활성화시켜주시면 됩니다.
   ```
  conda activate unsloth_env
  git clone https://github.com/joonyeongs/PersonaAgent.git
  cd PersonAgent  
  ``'

## 모델 훈련
모델은 script 폴더 내부에 있는 .sh 파일을 로컬 환경에 맞추어 수정하고, shell에 다음 명령을 입력하면 됩니다
'''
bash scripts/train_single_model.sh
'''

스크립트의 argument들은 다음과 같습니다:
'''
--dataset_dir 학습 데이터셋 경로 \
--beta DPO 최적화함수 베타 값  \
--output_dir 모델 저장경로 \
--epoch 에포크 수 \
--mbti 학습하기를 원하는 MBTI(저장 경로용) \
'''

또는 터미널에서 직접 실행시키셔도 됩니다.

'''
python trainer.py
--dataset_dir 학습 데이터셋 경로 \
--beta DPO 최적화함수 베타 값  \
--output_dir 모델 저장경로 \
--epoch 에포크 수 \
--mbti 학습하기를 원하는 MBTI(저장 경로용) \
'''


 
