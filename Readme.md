# Brain Tumor Classification 


Step - Data Preapration
     - Exploatory Data Analysis (EDA)
     - Data Argumentation
     - Data Preprocessing
     - Data Splitting
     - Model Building
     - Unfreezing and fine-tuning
     - Flask app(optional if time )


## STEP 1 - Data Preapration

We renamed all the file in the Yes directory as Y_1, Y_2... and No Directory as N_1, N_2...
Also we converted all the files into JPG format

## STEP 2 - Exploatory Data Analysis (EDA)
- first check if the data is imbalanced or not and then plot it using matplotlib pyplot library

## STEP 3 - Data Augmentation
as we can see that our dataset is quite imbalanced 155(yes img) and 98(no img)
so we will generate some extra images using `image data generator` in keras
for this we will be using transfer learning using VGG19
```python
data_gen = ImageDataGenerator(rotation_range=10,       # Randomly rotate the image by up to 10 degrees
                            width_shift_range=0.1,    # Randomly shift the image horizontally by 10%
                            height_shift_range=0.1,   # Randomly shift the image vertically by 10%
                            shear_range=0.1,          # Apply a shear transformation up to 10%
                            brightness_range=(0.3, 1.0),  # Adjust brightness between 30% to 100%
                            horizontal_flip=True,     # Randomly flip the image horizontally
                            vertical_flip=True,       # Randomly flip the image vertically
                            fill_mode='nearest')      # Fill missing pixels with nearest pixel value

     # Iterate over each image file in the specified directory (file_dir)
    for filename in os.listdir(file_dir):
        # Read the image file using OpenCV (cv2)
        image = cv2.imread(file_dir + '/' + filename)
        # Reshape the image to add an extra dimension, required by the data generator
        image = image.reshape((1,) + image.shape)
        # Create a prefix for saving the augmented images (remove the file extension)
        save_prefix = 'aug_' + filename[:-4]
        # Initialize a counter to track how many augmented images are generated
        i = 0
        # Generate augmented images using the data generator
        # 'flow' generates batches of augmented images from the original image
        for batch in data_gen.flow(x=image, 
                                   batch_size=1,               # Generate one augmented image at a time
                                   save_to_dir=save_to_dir,    # Directory to save augmented images
                                   save_prefix=save_prefix,    # Prefix for saving augmented image filenames
                                   save_format="jpg"):         # Save images in JPEG format
            i += 1  # Increment the counter each time an augmented image is generated
            # Stop after generating the specified number of augmented images (n_generated_samples)
            if i > n_generated_samples:
                break  # Exit the loop once enough augmented images are generated
```

now with this we have an augmented data 
```python
yes_path = "brain_tumor_dataset/yes"
no_path = "brain_tumor_dataset/no"

augmented_path = "augmented_data/"
augmented_data(file_dir=yes_path, n_generated_samples=6, save_to_dir=augmented_path+"yes")
augmented_data(file_dir=no_path, n_generated_samples=9, save_to_dir=augmented_path+"no")

```
after this we will again check if our data is balanced or not and then print it  
so now our data is balanced

 Number of samples: 2064
1085 Number of positive sample in percentage: 52.56782945736434%
979 Number of negative sample in percentage: 47.43217054263566%


## STEP 4 -Data Preprocessing
1. Cropping the Brain Tumor Region (crop_brain_tumor)
2. Loading and Augmenting Data (load_data)
3. Final Data Structure

convert BGR to GRAY format
then gaussian Blur
the we will apply teh threshold , erode , dilate adn lastly find the contours

making an image storing fucntion so that we can load the image function and try to plot some samples of the images


## STEP 5 - Data Splitting
split the data into train test validation for both  tumourous and non-tumorous
with the combination of 80-10-10

## STEP 6 - Model Building
- building the base model with VFF19 (pre-trained on the ImageNet dataset.)
- input_shape=(240, 240, 3) means the input images have a size of 240x240 pixels with 3 channels (RGB).
- weights='imagenet' initializes the model with weights pre-trained on the ImageNet dataset.

ALSO, loop sets all the layers in the base VGG19 model as non-trainable, meaning the weights of these layers won't be updated during training

- adding custom layers on top of the VGG19 model
        Total params: 140,946,370 (537.67 MB)
        Trainable params: 120,921,986 (461.28 MB)
        Non-trainable params: 20,024,384 (76.39 MB)

- then we complite the model  with loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']
- 
then we start to train the model on 50 steps per epoch for 50 epochs with batch size 32 and shuffled with all the callbacks such as early stopping model checkpoint 
- then we show plot performance

## STEP 7 - Increamental unfreezing and fine tuning
we start with loading the modelwith the top classification layers and the Retrieving and listing the names of the layers in the base model

```python 
base_model = VGG19(include_top=False, input_shape=(240,240,3))
base_model_layer_names = [layer.name for layer in base_model.layers] 
base_model_layer_names
```

then we build a custom model based on VGG19 by
- Loading VGG19
- Layer Names Extraction
- Adding Custom Layers
- Fine-tuning
- and lastly Model Summary

then we start with Training the Model with steps_per_epoch=75 and for epochs=50
followed by Saving Model Weights with Model Evaluation

## Unfrezzing the entire network
start with Model Architecture
    - VGG19 Base Model
    base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
    base_model_layer_names = [layer.name for layer in base_model.layers] 
    base_model_layer_names

    - Customizing the Model

    - Loading Pre-trained Weights
        model_03.load_weights('model_weights/vgg19_model_02.weights.h5')
    
    -  Model Compilation 
        sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model_03.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

Model Training 
    history_03 = model_03.fit(train_generator, steps_per_epoch=50, epochs=100, 
                          callbacks=[es, cp, lrr], validation_data=valid_generator)\

Atlast Plotting and Evaluation



