from mmap import MADV_PROTECT
from keras.preprocessing.text   import text_to_word_sequence, Tokenizer
from matplotlib.cbook import flatten
import numpy as np
import tensorflow as tf
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding
# text = '아버지가방에들어가신다'

# result= text_to_word_sequence(text)
# print("\n원문: \n", text)
# print("\n토큰화: \n", result)

docs=['먼저 텍스트의 각 단어를 토큰화 합니다.','텍스트의 단어로 토큰화 해야 딥러닝에서 인식됩니다.','토큰화를 한 결과는 딥러닝에서사용할 수 있습니다.']

token = Tokenizer()

# token.fit_on_texts(docs)

# print("\n단어 카운트\n", token.word_counts)

# print("\n문장카운트 : ",token.document_count)
# print("\n각 단어가 몇 개의 문장에 포함되어 있는가 : ",token.word_docs)
# print("\n각 단어에 매겨진 인덱스 값 : ",token.word_index)

print('\n======================================================\n')

docss=['너무 재밌네요','최고예요','참 잘 만든 영화예요','추천하고 싶은 영화 입니다.','한번 더 보고싶네요','글쎄요','별로예요', '생각보다 지루하네요','연기가 어색해요','재미없네요']
classes = array([1,1,1,1,1,0,0,0,0,0])

token.fit_on_texts(docss)
print(token.word_index)
x=token.texts_to_sequences(docss)
print('\n리뷰 텍스트, 토큰화 결과:\n',x)