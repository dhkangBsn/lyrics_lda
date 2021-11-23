import MeCab
import pandas as pd

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
            print(idx,lyrics[idx])
            word_morph_list = parse_sentence(lyrics[idx], target_tags, stop_word)
            word = [ word_morph[0] for word_morph in word_morph_list ]
            result.append(word)
    except:
        print(idx, '해당 인덱스에서 오류가 났습니다.')
    return result

df = pd.read_csv('../data/발라드.csv')
print(df.head())
lyrics = df['lyrics'].values
stop_word = ['것', '을', '겠', '은', '.', '는', ',']
word = extract_word_list(lyrics, target_tags, stop_word)
print(word)


