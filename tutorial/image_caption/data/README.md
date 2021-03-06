## [How to Prepare a Photo Caption Dataset for Training a Deep Learning Model](https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/)


환경
    
    Python 3 Scipy or 2
    Keras version 2.0 or higer
    scikit-learn, Pandas, Numpy, Matplotlib

1. Download the Flicker8K Dataset
   
   CPU에서 작업하기 위해서는 Flicker8K가 image captioning에 적합
   
   테스트용 데이터는 https://forms.illinois.edu/sec/1713398 에 요청
   
   Flickr8k_Dataset.zip
   Flickr8k_text.zip
   
2. How to Load Photographs

   파일명 : 99679241_adc853a5c0.jpg
   
   * step 1: load_img()
   
          from kears.preprocessing image import load_img
          image = load_img('99679241_adc853a5c0.jpg')
   
   * step 2: Numpy array로 변환
   
          from keras.preprocessing.image import img_to_array
          image = img_to_array(image)
          
   * step 3: pre-defined model(VGG) 이용
   
          from keras.applications.vgg16 import preprocess_input
          image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
          image = preprocess.input(image)
          
          image = load_img('99679241_adc853a5c0.jpg', target_size=(224,224))
          
          image_id = filename.split('.')[0]

    *part1_load_data.py*

3. Pre-Calculate Photo Features

    Pre-trained 모델을 이용한 방법으로 진행
    
    * step 1: Load VGG Model
    
          from keras.applications.vgg16 import VGG16
          in_layer = Input(shape=(224,224,3))
          model = VGG16(include_top=False, input_tensor=in_layer, pooling='avg')
    
    * step 2: Dense Output Layer remove from model
    
          include_top = False

    * step 3: call predict()
    
    
    *part2_pre_calculate.py*
    
    
4. Load Description

    Flickr8K.token.txt 파일에서 image에 대한 설명이 있음(1사진에 여러 개의 설명)
    
    * step 1: Annotation file('Flickr8K.token.txt') load 
    
            def load_doc(filename):
                file = open(filename, 'r')
                text = file.read()
                file.close()
                return text

    * step 2: split each line
        
            tokens = line.split()
            image_id, image_desc = tokens[0], tokens[1:]

    * step 3: remove filename extension
    
            image_id = image_id.split('.')[0]

    * step 4: convert description tokens back to string
        
            image_desc = ' '.join(image_desc)

    *part3_load_desc.py*


5. Prepare Description Text

    text 전처리 : 모든 token을 소문자, 구두점 삭제, 's/ 'a'등의 작은 문자 삭제
    
            def clean_descriptions(descriptions):
                table = str.maketrans('','', string.punctuation)
                for key, desc in descriptions.items():
                    desc = desc.split()
                    desc = [word.lower() for word in desc]
                    desc = [w.translate(table) for w in desc]
                    desc = [word for word in desc if len(word) >1]
                    descpriptions[key] = ' '.join(desc)

    *part4_parpare_desc.py*


6. Whole Description Sequence Model

    사진에서 특징들을 추출한 후 단어와 Mapping으로 나열한 형태로
    Encoder-Decorder RNN 모델과는 다른형태
    
    * step 1: 전처리 이미지를 memory load
    
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
                    decriptions[image_id] = ' '.join(image_desc)
                return descriptions
            descriptions = load_clean_descriptions('descriptions.txt')

    * step 2: extract text
        
            desc_text = list(descriptions.values())

    * step 3: Mapping Vocabulary to an integer
    
            from kears.preprocessing.text import Tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(desc_text)
            vocab_size = len(tokenizer.word_index) + 1

    * step 4: integer encode descriptions
    
            sequences = tokenizer.texts_to_sequences(desc_text)

    * step 5: Pad the sequence(모든 encoded sequence들이 같은 길이를 가져야 하기 때문)
    
            from keras.preprocessing.sequence import pad_sequences
            max_len = max(len(s) for s in sequences)
            padded = padd_sequences(sequences, maxlen=max_length, padding='post')

    * step 6: One hot encode
    
            from keras.utils import to_categorical
            y = to_categorical(padded, num_classes=vocab_size)
            

    *part5_WholeDescSeq_model.py*

7. Word by Word Model

    참고 모델: [A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
    
    * step 1: split into input and output pair
        
            in_seq, out_seq = seq[:i], seq[i]
            
    * step 2: pad input sequence
        
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

    * step 3: output word encode one-hot
            
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            

    *part6_wordxword_model.py*
    
8. Progressive Loading 

    Keras supports progressively loaded datasets by using the fit_generator() function on the model.
    
    *part7_progressive_Loading.py*
    

## [How to Develop a Deep Learning Photo Caption Generator from Scratch](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)

1. Loading Data

    How to Prepare a Photo Caption Dataset for Training a Deep Learning Model에서 이미 정의된 내용
    다만, photo데이터 extract할 때, VGG 모델의 마지막 layer는 classification부분이기 때문에 제외 
        -> model.pop()을 통해서 re-strcuture 

2. Defining the Model
    
    Merge-Model 기준으로 작성
    
    참고 논문: [Where to put the Image in an Image Caption Generator](https://arxiv.org/abs/1703.09137), [What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?](https://arxiv.org/abs/1708.02043)
    
    word -> RNN  -> FF
            image->

    * Photo Feautre Extractor : VGG16모델 사용
    
    * Sequence Processor : LSTM사용
    
    * Decoder : Merge-model을 통한 최종 예측
    
            def define_mode(vocab_sizze, max_length):
                # feature extractor model
                inputs1 = Input(shape(4096,))
                fe1 = Dropout(0.5)(inputs1)
                fe2 = Dense(256, activation='relu')(fe1)
                # sequence model
                inputs2 = Input(shape(max_length,))
                se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
                se2 = Dropout(0.5)(se1)
                se3 = LSTM(256)(se2)
                # decoder model
                decoder1 = add([fe2, se3])
                decoder2 = Dense(256, activation='relu')(decoder1)
                outputs = Dense(vocab_size, activation='softmax')(decoder2)
                # tie it together [image, seq] [word]
                model = Model(inputs=[inputs1, inputs2], outputs=outputs)
                model.compile(loss='categorical_crossentropy', optimizer='adam')
                # summarize model
                print(model.summary())
                plot_model(model, to_file='model.png', show_shapes=True)
                return model
                
                
    
3. Fitting the Model

    ModelCheckpoint in Keras and specifying it to monitor the minimum loss on the validation dataset and save the model to a file that has both the training and validation loss in the filename.
    
        # define checkpoint callback
        filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        
        # fit model
        model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test],ytest))
        

4. Complete Example
    
    *part8_prepapre_data.py*
    *part8_prepapre_text.py*
    *part8_loading_data.py*
    *part8_deeplearning_model.py*
