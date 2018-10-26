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



