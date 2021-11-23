import requests
import pandas as pd

STOP_URL = "https://gist.githubusercontent.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a/raw/4f7a635040442a995568270ac8156448f2d1f0cb/stopwords-ko.txt"
stop_words = requests.get(STOP_URL).text.split("\n")
print(stop_words)