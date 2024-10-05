# 👮 LAPD: Large language model Alignment through Persona Dynamics

📢 2024년 여름학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다 \
📢 2024년 전기 고려대학교 산학협력단의 일경험사업 지원으로 진행한 프로젝트입니다

## 팀원
| 팀원                            | 역할                                       |
| ----------------------------- | ---------------------------------------- |
| [박서현](https://github.com/emiliebell) | Data Generation, Result Analysis, Statistical Analysis, Figure Design   |
| [성준영](https://github.com/joonyeongs)      |   Data Generation, Benchmark Evaluation, Result Analysis, Paper Writing  |
| [이종훈](https://github.com/jhoonjwa)     |  Model training, Model MBTI evaluation, Result Analysis, Data Generation, Paper Writing |
| [황정현](https://github.com/imjunghyunee)        | Data Generation, Result Analysis, Paper Writing             |


## 소개
😊 특정 MBTI 특성을 갖도록 LLM Fine-Tuning \
👮‍♂️ 각 모델이 인간의 가치와 윤리에 얼마나 잘 부합하는지에 대한 Qualitative & Quantitative Analysis 

최근 LLM들은 다양한 벤치마크에서 인간의 지능에 근접한, 어떤 분야에서는 뛰어넘는 성능을 보여주고 있습니다. \
❔그러면, LLM도 인간처럼 성격을 가지고, 그 성격을 바꿀 수 있을까요? \
❔LLM의 성격이 바뀐다면, 그 답변 양상은 어떻게 바뀔까요? \
❔달라진 성격을 가진 LLM이 과연 인간 사회에 해를 끼치지 않고 잘 어울릴 수 있을까요? \

이에 대한 답을 찾기 위해, MBTI로 대표되는 persona를 모델에게 학습시켜, MBTI별 모델의 답변 양상을 관찰하였습니다.
본 프로젝트에서는 LLM이 특정 페르소나(MBTI)로 fine-tuning 되었을 때, **모델의 안전성과 사회성에 초점을 맞춰 AI Alignment**를 분석하고, 이를 위해 persona를 활용한 📛Red-Teaming📛 기법을 제안합니다. .  

더 많은 정보를 얻고싶으시다면, [AIKU 노션](https://www.notion.so/aiku/Mamihlapinatap-ai-d0100f9c85424342bd47a2c496ebe25e)에 방문하셔서 확인해 주세요! \
저희 프로젝트는 논문으로 이어졌으며, 해당 논문은 🏅 LLM Alignment with Persona Dynamics.pdf 🏅에서 확인하실 수 있습니다.

## 방법론

이를 위해 <ins>LLaMA-3 Instruct 8B</ins> 모델을 baseline 모델로 설정하고, <ins>DPO(Direct Preference Optimization)</ins>를 이용해 각각의 MBTI에 맞게 fine-tuning을 진행했습니다. \
모델 학습은 자체 제작한 데이터셋으로, DPO(Direct Preferance Optimization)으로 학습하였으며, 

### 1. 데이터 생성 ###
<img src="src/5.jpg" alt="Description of image" width="500" height="300"> \
파이프라인은 다음과 같습니다: \
(1) MBTI 관련 데이터 웹 크롤링 + 상황 데이터 수집 (예. 카페에서 수다, 직장에서 상사와 얘기, 놀이공원 방문 등) \
(2) (1)에서 수집한 데이터 기반으로 각각의 MBTI별 dialogue 생성(Claude API 활용) \
(3) (2)에서 생성한 dialogue를 "User" - "MBTI"로 파싱 \
(4) 각각의 paired data에 대한 rejected response 생성 

### 2. 모델 학습 ###
<img src="src/7.jpg" alt="Description of image" width="500" height="300">
<img src="src/6.jpg" alt="Description of image" width="500" height="300"> 

- 제한된 자원으로 인해 Paremeter-Efficient Fine-tuning의 일종인 LoRA(Low Rank Adaptation) 활용
- 목표로 하는 성격유형을 LLM에 주입하기 위해 DPO 사용


### 3. 벤치마크 평가 ###
<img src="src/8.jpg" alt="Description of image" width="500" height="300"> 

- [SocialQA](https://arxiv.org/pdf/1904.09728), [Beavertails](https://arxiv.org/abs/2307.04657), [EQ-Bench](https://arxiv.org/abs/2312.06281)와 같은 LLM의 사회성, 공감능력과 안전성을 평가하는 벤치마크를 활용

### 4. Red-Teaming 기법 제안 ###
<img src="src/4.jpg" alt="Description of image" width="500" height="300">

- Red-teaming은 모델 출력에서 잠재적인 취약성과 편견 및 부적절한 동작을 식별하는 것
- 본 프로젝트의 분석으로 통하여, 특정 성격유형의 특징으로 부적절한 답변을 유발할 수 있다는 사실을 발견
- 이를 활용하여 잠재적인 jailbreaking scenario 탐지 -> 보안 조치 마련 가능


## 환경 설정
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
cd PersonaAgent  
```

## 사용 방법

**모델 학습**
모델은 script 폴더 내부에 있는 .sh 파일을 로컬 환경에 맞추어 수정하고, shell에 다음 명령을 입력하면 됩니다

```
bash scripts/train_single_model.sh
    --dataset_dir data/generated_data/entj_paired_data.json \
    --beta DPO 0.1  \
    --output_dir models/entj \
    --epoch 4 \
    --mbti ENTJ \
```


## 예시 결과
<img src="src/1.jpg" alt="Description of image" width="500" height="300">
<img src="src/2.jpg" alt="Description of image" width="500" height="300">
<img src="src/3.jpg" alt="Description of image" width="500" height="300">



 
