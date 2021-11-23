
import MeCab
import pandas as pd
import gensim
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

m = MeCab.Tagger()

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

def parse_sentence(sentence, target_tags, stop_word):
    result = m.parse(sentence)
    temp = result.split('\n')
    temp_2 = [ sentence.split('\t') for sentence in temp]
    words = [ sentence[0] for sentence in temp_2 ]
    morphs = [ sentence[1].split(',')[0]
               for sentence in temp_2
               if len(sentence) > 1]
    morphs = [ morph for morph in morphs if morph in target_tags ]
    words = words[:len(morphs)]



    word_morph = [ (word,morph)
                   for morph, word in zip(morphs, words)
                   if word not in stop_word ]
    return word_morph

def extract_word_list(lyrics, target_tags, stop_word):
    result = []
    try:
        for idx in range(len(lyrics)):
            word_morph_list = parse_sentence(lyrics[idx], target_tags, stop_word)
            word = [ word_morph[0] for word_morph in word_morph_list if len(word_morph[0]) > 1]
            result.append(word)
    except:
        print(idx, '해당 인덱스에서 오류가 났습니다.')
    return result

df = pd.read_csv('./data/발라드.csv')
print(df.head())
lyrics = df['lyrics'].values
stop_word = ['것', '을', '겠', '은', '.', '는', ',']
word = extract_word_list(lyrics, target_tags, stop_word)

def make_bigram(word):
    return gensim.models.Phrases(word, min_count=5, threshold=100)

def make_trigram(word):
    bigram = gensim.models.Phrases(word, min_count=5, threshold=100)
    return gensim.models.Phrases(bigram[word], threshold=100)
#print(bigram)
def make_trigram_list(word, bigram_mod, trigram_mod):
    trigram_list = []
    for idx in range(len(word)):
        trigram_list.append(trigram_mod[bigram_mod[word[idx]]])
    return trigram_list
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = make_bigram(word)
trigram_mod = make_trigram(word)
trigram_list = make_trigram_list(word, bigram_mod, trigram_mod)
print(trigram_list[0])

# Create Dictionary and corpus for topic modeling
id2word = corpora.Dictionary(trigram_list)
print(id2word[0])

corpus = [id2word.doc2bow(text) for text in trigram_list]
#print(corpus)

# View
print('corpus', corpus[:1])
print('id2word', id2word[corpus[0][0][0]])

#
temp = [[(id, id2word[id], freq) for id, freq in cp][:10] for cp in corpus[:1]]
print('상위 10개 단어', temp)

# gensim.models.ldamodel
#class gensim.models.ldamodel.LdaModel(corpus=None,
#                                     num_topics=100,
#                                     id2word=None,
#                                     distributed=False,
#                                     chunksize=2000,
#                                     passes=1,
#                                     update_every=1,
#                                     alpha='symmetric',
#                                     eta=None,
#                                     decay=0.5,
#                                     offset=1.0,
#                                     eval_every=10,
#                                     iterations=50,
#                                     gamma_threshold=0.001,
#                                     minimum_probability=0.01,
#                                     random_state=None,
#                                     ns_conf=None,
#                                     minimum_phi_value=0.01,
#                                     per_word_topics=False,
#                                    callbacks=None,
#                                     dtype=<class 'numpy.float32'>)

print(id2word)
NUM_TOPIC = 3
lda_model = gensim.models.ldamodel.LdaModel(iterations=200,
                                            corpus=corpus,
                                            id2word=id2word,
                                            num_topics=NUM_TOPIC,  #만약에 토픽이 8개라면, 그러면 그 토픽은 무엇이니?
                                            random_state=100,
                                            chunksize=400,
                                            passes=100,  # 중복된 토픽이 나오는 경우, 에폭을 늘려야한다.
                                            alpha='auto',
                                            per_word_topics=True
                                            )

# Print the Keyword in the 10 topics
# ldamodel이 정한 토픽중 앞쪽 순서 10개의 토픽에 해당되는 키워드들입니다.
# 각 키워드들에는 가중치가 정해져있습니다.
# 이 가중치들을 바탕으로 문서의 토픽을 분류합니다.
pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]
print(doc_lda)

#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)


#vis