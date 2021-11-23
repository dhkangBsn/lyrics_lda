import pandas as pd
import re
from pprint import pprint
import gensim
import tqdm
# drop: 필요없는 column 제외.
import requests
from gensim.utils import simple_preprocess


df = pd.read_csv('https://gist.githubusercontent.com/ArtemisDicoTiar/7d91f779b2a9a7485009cb3f129fd711/raw/3a064717024895612f8684dbf8b7d67f5e70cae0/ko_news.csv')\
        .drop(columns=['Unnamed: 0', 'f_name'])\
        .rename(columns={'idx': 'target', 'txt': 'content'})

# Convert to list
# 데이터프레임의 함수를 사용해서 전처리를 하기에는 복잡하다
# 그래서 리스트로 바꿔서 전처리를 하겠습니다
data = df.content.values.tolist()

# data: List[str] - str=기사.
for text in data[:3]:
  print(text)
  print("---")

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
print(data)

# 특수 따옴표 제거
data = [re.sub(" “", "", sent) for sent in data]  # 시작 따옴표
data = [re.sub("” ", "", sent) for sent in data]  # 마침 따옴표

# Remove new line characters
# \n
data = [re.sub('\s+', ' ', sent) for sent in data]
pprint(data[:1])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        # deacc=True removes punctuations
        # gensim's simple_preprocess does the job that we desire.

data_words = list(sent_to_words(data))

# data_words = List[str] -> List[List[str]]
print(data_words[:1])



# Creating Bi-gram, Tri-gram models
# Bi-gram : two words frequently occuring together in the document.
# Tri-gram: same as bigram but three words.
# 가장 자주 쓰이는 단어 두개, 세개 혹은 그 이상을 묶어주는 모델을 만들어 봅시다.
# 다행히도 gensim이 해당 모델을 제공해줍니다.
# gensim의 Pharases 모델을 사용합시다.
# 여기서 말하는 단어쌍은 다음과 같은 겁니다.
# happy + birthday -> happy_birthday
# oil + leak -> oil_leak
# maryland + college + park -> maryland_college_park
# 아래 결과를 보면 nntp_posting_host로 세개의 단어가 뭉쳐 있는 것을 볼 수 있다.
# 이렇게 단어를 뭉쳐서 collection 형태로 표현하는 이유는 다음과 같은 현상 때문입니다.
# Uni-grams
# - topic1 -scuba,water,vapor,diving
# - topic2 -dioxide,plants, green, carbon
# Bi-gram topics
# - topic1 - scuba diving, water vapor
# - topic2 - green plants, carbon dioxide

# Build the bigram and trigram models
# gensim's Phases model can implement the bi-gram, tri-gram and more.
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example

# print(bigram_mod[data_words[0]])
# print(data_words[0])
# print(trigram_mod[bigram_mod[data_words[0]]])
# print(data_words[0])
print(trigram_mod[bigram_mod[data_words[0]]])


target_tags = [
    'NNG',  # 일반 명사
    'NNP',  # 고유 명사
    'NNB',  # 의존 명사
    'NR',  # 수사
    'NP',  # 대명사
    'VV',  # 동사
    'VA',  # 형용사
    'MAG',  # 일반 부사
    'MAJ',  # 접속 부사
]

f = requests.get('https://gist.githubusercontent.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a/raw/4f7a635040442a995568270ac8156448f2d1f0cb/stopwords-ko.txt')
stop_words = f.content.decode("utf-8").split('\n')
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# 두 단어로 구성되는 연어를 찾아 토큰을 재구성
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# 세 단어로 구성되는 연어를 찾아 토큰을 재구성
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# 표제어 추출은 아님.
# # 원형으로 정규화.
# def lemmatization(texts, allowed_postags=target_tags):
#     results = list()
#     for text in texts:
#         # 품사추출만 진행.
#         results.append([s for s, t in kkma.pos(' '.join(text)) if t in allowed_postags and len(s) > 1])
#     return results





def filter_pos(texts, allowed_postags=target_tags):
    results = list()
    for text in tqdm(texts):
        # 품사추출만 진행.
        results.append([s for s, t in kkma.pos(' '.join(text)) if t in allowed_postags and len(s) > 1])
    return results



