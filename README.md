# 📝 BART 모델을 사용한 문서번역 및 요약



# 개요 
우리는 새로운 정보를 얻거나, 기존의 정보들에 좀 더 심화된 과정에 대한 연구를 시작하기 위해 많은 검색을 합니다.

\
이 과정에서 수 많은 논문들을 마주치게 되는데, 다양한 분야의 연구 및 전문적인 정보 습득을 위한

관점에 있어서 논문에 대한 분석은 필수 요소로 자리 잡았습니다.

\
**딥러닝과 머신러닝 논문**을 읽는 과정에서 매끄럽지 않게 해석되는 어려움,

더 많은 논문을 읽는데 있어 소비되는 시간 문제 또한 해결할 수 있는 방안을 찾고자 하였습니다.

\
이 모델은 원문으로 쓰여있는 많은 논문들을 해석하는 점에 있어 능숙하지 않은 사람들에게 

많은 원문 논문을 좀 더 쉽고 편리하게 볼 수 있도록 했으며

\
주 타겟을 **IT 관련 논문 번역 및 요약**으로 정했습니다.

\
문서번역과 요약을 합한 파이프라인을 구축하여 영어원문을 넣으면 output으로 번역된 한글원문과

나온 한글원문이 다시 input으로 들어가 output으로 한글요약문이 나오게 설계하였습니다.

# 완성된 모델


<br><br>
# Dataset
- 문서번역 AI Hub 한국어-영어 번역 말뭉치(기술과학) : https://aihub.or.kr/aidata/30719 
- 문서요약 AI Hub 논문자료 요약(한글) : https://aihub.or.kr/aidata/30712 

총 데이터 개수

| Data                | # total size   |
| ------------        | -------------: |
| EN to KR (ICT part) |        350,000 |
| Text to Summary     |        180,000 |

# BART

BART는 Bidirectional Auto-Regressive Transformer의 약자이며
BERT와 GPT를 하나로 합친 형태로, 기존 seq2seq transformer 모델을\
새로운 Pre-training objective를 통해 학습하여 하나로 합친 모델입니다.

BART 모델이 나오며 여러 자연어 벤치마크에서 sota를 달성한 역사를 보고\
공통된 모델로 BART를 선정하게 되었습니다.

# Model Used

| Model               | Purpose        |
| ------------        | -------------: |
| mBART 50 (large)    |    translation |
| KoBART              |  Summarization |


## mBART 50_large (Translate thesis EN to KR)

mBART는 50개의 언어로 구성된 말뭉치를 활용하는 모델으로써\
그 중 tokenizer에 영어(en-XX)와 한국어(ko-KR)를 적용하여 사용했습니다.\
mBART는 BART와 유사하게 사전학습 단계에서 원본 문장을 고의로 훼손하여\
입력 데이터를 만든 후 이를 원본 문장으로 복원하는 작업을 학습합니다.\
\
이때, 원본 문장을 훼손하는 방법(Noising Scheme)으로는 연속된 단어를 하나의
[MASK]토큰으로 치환하는 Text Infilling 방법과\
입력 내에서 문장의 순서를 바꾸는 Sentence Permutation방법 두 가지를 활용합니다.

## KoBART (Summarize KR thesis)

KoBART는 SKT가 개발한 모델로서 기존 BART 모델에서 사용된 Text Infilling 노이즈 함수를 사용해\
약 40GB 이상의 한국어 텍스트에 대해 학습한 한국어 encoder-decoder 언어 모델입니다.

# DATA PRE-PROCESSING
- AI Hub의 데이터는 철저한 품질 관리가 이루어진 데이터로써 특별한 전처리 과정이 필요치 않았습니다.
- 
## Translate

- 한국어-영어 번역 말뭉치 중 기술과학 분야 35만개를 다운로드 했으나
학습 과정에서의 소요시간을 줄이기 위해 그 중 빅데이터와 컴퓨터 카테고리의 데이터\
총 15만5천개 데이터를 사용했습니다.

## Summarization

- 논문자료 요약(한글) 중 학술논문(전체요약) 18만개의 데이터를 json파일로써 다운로드 받았고\
데이터 중 'text'와 'summary' 즉 '원문'과 '요약문'만 가져와 사용했습니다. 

- 기존 BART모델에 노이즈를 만들어내는 Noising Scheme 방법이 있지만
AI Hub의 데이터는 매우 깔끔히 정돈된 데이터여서\
노이즈 추가가 필요하다 판단, Selenium을 통해 자동으로 1만개의 영어문장을 넣고\
삼성 SR Translation(https://translate.samsung.com/) 번역기에 돌린 번역문(한글)을 크롤링해서 노이즈 데이터를 획득하였습니다.\
획득한 노이즈 데이터를 기존 데이터 14만개 + 노이즈 데이터 1만개(번역 후 나온 한글) 총 15만개를 KoBART 모델로 학습을 진행했습니다.


# Modeling

- Huggingface API (tokenizer, model)을 download 및 import
```python
!pip install git+https://github.com/huggingface/datasets.git@master
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ko_KR")
```
- 전처리가 끝난 데이터 zip을 이용해 하나의 data 생성

```python
data = []
with open("/content/en.csv") as f1, open("/content/ko.csv") as f2:
    for src, tgt in zip(f1, f2):
      data.append(
          {
                  "ko": tgt.strip(),
                  "en": src.strip()
          }
      )
print(f'total size of data is {len(data)}')
```

- 딕셔너리 형태로 전환 후 토큰화 및 정수화 과정을 진행

```python
def preprocess_function(examples):
    inputs = [doc for doc in examples["en"]] 
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True) 
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["ko"], max_length=1024, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

```python
tokenized_data = data.map(preprocess_function, batched=True)
```

- 데이터셋을 pre-trained된 모델로 학습합니다.

----위 방식과 비슷한 방식으로 문서요약도 진행됩니다 (해당 ipynb파일 참조)----


# Result

## translation 

Colab pro+ 기준 할당된 GPU보다 학습시 많은 GPU를 사용하게 되면서 다운되는 현상이 일어나\
Batch_size를 8, epoch을 15로 설정하고 학습시켰습니다.

| Epoch               | Training Loss  | Validation Loss|
| -------------:      | -------------: | -------------: |
|1	                  |  0.843600      |0.789460        |
|2	                  |  0.684200	     |0.722583        |
|3	                  |  0.546200	     |0.717286        |
|4	                  |  0.431400	     |0.740622        | 
|5	                  |  0.335800	     |0.791957        |
|6	                  |  0.272400	     |0.853283        |
|7	                  |  0.197200	     |0.931563        |
|8	                  |  0.143800	     |0.993216        |
|9	                  |  0.111800	     |1.042014        |
|10                   |  0.083300	     |1.086850        |

그러나 5 epochs부터 Training Loss는 꾸준히 줄어들지만 Validation Loss가 늘어나는
Overfit 현상이 일어나서 10 epochs에서 학습을 중단하고 5 epochs에 체크포인트로 저장되어있는\
모델과 tokenizer로 번역기를 돌려보았습니다.


영어 원문
`'Build a Machine Learning web application from scratch in Python with Streamlit. We use real world data to build a machine learning model. In the first part you learn how we analyze the data and build our model, and in the second part we build the web app using streamlit.'`

한글 번역문
`'Streamlit으로 Python에서부터 머신러닝 웹 애플리케이션을 구체화 시켜 놓는다. 실세계 데이터를 이용해 머신러닝 모델을 구축한다. 첫 번째 파트에서는 데이터 분석 방법을 배우고 머신러닝 모델을 구축하는 두 번째 파트에서는  streamlit을 이용해 머신러닝을 이용한 웹 앱을 구축한다.'`

- 낮지 않은 Validation Loss에도 불구하고 좋은 성능을 보입니다.
- 또한 IT 카테고리의 데이터를 학습시킨 결과 '머신러닝'등 관련 전문용어에 대한 해석도 훌륭히 해내는 모습을 보입니다.

## summarization

번역에서 GPU 메모리 이슈를 겪었고 데이터의 양 또한 더 많아서 번역과 동일하게
Batch_size를 8, epoch을 15로 설정하고 학습시켰습니다.

| Epoch               | Training Loss  | Validation Loss|
| -------------:      | -------------: | -------------: |
|1	                  |  2.122000      | 	2.020537      |
|2	                  |  1.966700	     |  1.986229      |
|3	                  |  1.912900	     |  1.972917      |
|4	                  |  1.765300      | 	1.974346      | 
|5	                  |  1.757700      | 	1.978424      |
|6	                  |  1.621700      | 	1.988834      |
|7	                  |  1.667400      | 	1.998288      |
|8	                  |  1.625300	     |  2.010657      |
|9	                  |  1.558700	     |  2.018661      |
|10                   |  1.575400	     |  2.025263      |


그러나 학습이 시작되었지만 처음부터 Training, Validation Loss의 값이 2이상으로 나왔고
학습이 진행될수록 loss가 낮아졌으나 5epoch 이후부터 Validation Loss값이 올라가는
Overfit 현상이 나타났습니다.

결국 위와 마찬가지로 10 epochs에서 학습을 중단하고 5 epochs에 체크포인트로 저장되어있는\
모델과 tokenizer로 요약모델을 돌려보았습니다.

- 영어에서 번역된 한글 원문
`'언어로 된 텍스트를 네트워크 분석 대상으로 하여, 그 내용을 분석하는 방법을 언어 네트워크  분석(language network  analysis)이라고 한다. 
언어 텍스트로 표프된 메시지에 내재된 다양한 특성들을 나타내는 개념들을 추출하고, 그들 간에 형성되는 의미윁 관계의 속성들을 파악하고자 할 때 언어 네트워크 분석 방법을 사용하면 매우 유용하다. 
일반윁으로 언어 텍스트의 특성을 나타내는 개념은 키워드(딐는 단어)로 표프되며, 명사형태의 단어, 특정한 범주에 속하는 단어, 감성을 나타내는 단어 등으로 나타난다.
방법론으로 보면, 언어 네트워크 분석은 내용분석(content analysis) 방법의 범주에 해당된다고 볼 수 있다. 
위통윁인 내용분석은 연구논문, 언론기사, 인터뷰자료, 기록자료 등과 같은 언어 텍스트에서 특정한 개념들(윀자, 년도, 주제 등의 특성)이 등장하는 경향을 빈도와 같은 통계윁  데이터로  파악하는  방법이다. 
반면에, 언어 네트워크 분석은 언어 텍스트로부터 특정한 개념들의 관계를 파악하고, 이것을 네트워크로 구성하여, 계량윁인 특성을 분석하는 것까지 확대된 내용분석 방법이다. 
이러한 개념들 간의 관계를 언어 네트워크(language network)로 표프한다'`

- 한글 요약문
`" 언어 텍스트로 표프된 메시지에 내재된 다양한 특성들을 나타내는 개념들을 추출하고, 그들 간에 형성되는 의미윁 관계의 속성들을 파악하고자 할 때 언어 네트워크 분석 방법을 사용하면 아주 유용하다.  일반윁으로 언어 텍스트의 특성을 나타내는 개념은 키워드로 표프되며, 명사형태의 단어, 특정한 범주에 속하는 단어, 감성을 나타내는 단어 등으로 나타난다." `

- 처음 보았을 때는 문장이 자연스럽고 획기적으로 단어와 문장의 수가 줄어들어 요약에 성공한 것으로 간주했으나
원문의 내용을 요약한 것이 아닌 약 150음절이 넘어가면 문장을 자르고 끝내버려 그 이후의 글은 생략해버리는 결과가 나왔습니다.

# Pipline

- 추후 추가 예정

# Web translator application

- pyscript를 이용한 웹 번역기 사이트 연구

- 추후 추가 예정


# 마치며...

일주일동안 진행되었던 프로젝트 (05/25 ~ 06/02)

첫째날에 주제 선정에서 바로 논문 번역 및 요약 아이디어가 나왔고

그 이후 데이터 선정부터 파이프라인 구축 및 웹 개발까지

주말,공휴일까지 반납하며 열심히 해주신 조원분들

정말 수고 많으셨습니다 ヽ（≧□≦）ノ




## 참고 자료

- https://www.ajunews.com/view/20201210114639936
- https://dladustn95.github.io/nlp/BART_paper_review/
- https://github.com/SKT-AI/KoBART#how-to-install
- https://chloelab.tistory.com/34
- https://deepkerry.tistory.com/32
- https://dladustn95.github.io/nlp/BART_paper_review/
- https://dacon.io/competitions/official/235829/codeshare/4047
- https://www.koreascience.or.kr/article/CFKO202130060780846.pdf
- https://www.koreascience.or.kr/article/JAKO202116954611776.pdf
- https://jiwunghyun.medium.com/acl-2020-bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-7a0ae37109dc
