import MeCab
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
def parse_sentence(sentence, target_tags):
    result = m.parse(sentence)
    temp = result.split('\n')
    temp_2 = [ sentence.split('\t') for sentence in temp]
    words = [ sentence[0] for sentence in temp_2 ]
    morphs = [ sentence[1].split(',')[0]
               for sentence in temp_2
               if len(sentence) > 1]
    morphs = [ morph for morph in morphs if morph in target_tags ]
    words = words[:len(morphs)]
    word_morph = { word : morph for morph, word in zip(morphs, words)}
    return words, morphs, word_morph

print(parse_sentence("아버지가방에 들어가신다.", target_tags))