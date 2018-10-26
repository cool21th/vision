from os import listdir
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def load_photos(directory):
    images = dict()
    for name in listdir(directory):
        filename = directory + '/' +name
        image = load_img(filename, target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        image_id = name.split('.')[0]
        images[image_id] = image

    return images

# load images
directory = 'Flicker8K_Dataset'
images = load_photos(directory)
print("Loaded Images: %d" % len(images))
