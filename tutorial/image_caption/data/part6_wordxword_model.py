from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_clean_descriptions(filename):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]

        descriptions[image_id] = ' '.join(image_desc)
    return descriptions

descriptions = load_clean_descriptions('descriptions.txt')
print('loaded %d' % len(descriptions))

desc_text = list(descriptions.values())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) +1
print('Vocabulary size : %d' % vocab_size)

sequences = tokenizer.texts_to_sequences(desc_text)
max_length = max(len(s) for s in sequences)
print('Description Length : %d' % max_length)

X, y   = list(), list()
for img_no, seq in enumerate(sequences):
    for i in range(1,len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

        X.append(in_seq)
        y.append(out_seq)

X, y = array(X), array(y)
print(X.shape)
print(y.shape)
