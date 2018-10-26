from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.utils import to_categorical

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_clean_description(filename):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]

        descriptions[image_id] = ' '.join(image_desc)
    return descriptions

descriptions = load_clean_description('descriptions.txt')
print('Loaded %d' % (len(descriptions)))

desc_text = list(descriptions.values())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) +1
print('Vocabulary Size: %d' % vocab_size)

sequences = tokenizer.texts_to_sequences(desc_text)
max_length = max(len(s) for s  in sequences)
print('Description Length : %d' % max_length)

padded = pad_sequences(sequences, maxlen=max_length, padding='post')

y = to_categorical(padded, num_classes=vocab_size)
y = y.reshape((len(descriptions), max_length, vocab_size))
print(y.shape)
