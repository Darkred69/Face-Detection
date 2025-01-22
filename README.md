# Face Detection using a Custom VGG16 model
## Overview

This Face Detection algorithm is built in the understanding as a combination of a classification and regression model. Where the Classification here is to classify weather this is a face or not, and the regression is the 2 points of x and y that will built a rectangle around the face. 

## Workflow
The pojects contains 9 parts.
- Setup and Get Data
- Review Dataset and Build an Image Loading Function
- Partition Unaugmented Data
- Apply AlbumAugmentation for Transformation of Images and Labels
- Prepare the Labels
- Combine Label and Image Samples
- Built the Deep Learning Model
- Plot the performanceperformance
- Making Prediction on Test Set
- Real-time Prediction

## 1. Setup and Get Data
- The data is collected using the local machine's camera, it `takes 30 images` every `5 seconds`
- The images is saved as a **.jpg** having **640x480 pixels RGB**
- Uses `labelme` to draw the **rectangles** where alocates the face.

**Note_1:** All the picture will be store in `data/images` and all the labels is store in `data/labels`

**Note_2:** If you plan to run this on colab, get the images on your local machine first then compress the `data` into a zip file than run the code that extract the zip file.

## 2. Review the dataset
- Limit GPU Growth to avoid unessary errors
- Load an image as a `Tensorflow DataFrame`
- Visualize the images on Python as a batch

## 3. Partition Unaugmented Data
- Partition the data into Train, Test and Validation dataset (70-20-10)
- Start with partitioning the of `data/images` using `train_test_split`
- Add the corresponding labels from the `data/labels` sinces the images and labels differences is the extension (`jpg` and `jsosn`)

**Note:** This requries to create a `train`, `test` and `val` dir in the `data` dir. And inside each should have their own `images` and `labels` dir.

## 4. Apply AlbumAugmentation for Transformation of Images and Labels
- Some information about AlbumAugmentation
  - It contains rich Augmentation technique, including croping, brightness, contrasts, flips and RGB shifting
  - High Performance when it is optimized for speed and efficentcy and also ultilizes Numpy and other low-level libraries
  - Perservation of Label Integrity, where it ensures associated labels (e.g bounding boxes and masks) are accurately transformed alongside images, maintaining the integrity of the data.
- We create an augmentor, where we want to do random crop, horizontal flips, RandomBrightnessContrast, RandomGamma, RGB shift and Vertical flips.
- For the labels, we uses `albumentations` method, where we normalize our x_max, y_max and x_min, y_min
- We also decrease the resolution from the original image down to **120x120 pixel**
- We also test and tries to run the augmentor as well as visualize the result
- Than we apply the augmentation for the train, test and validation dataset, each **60 times** and also map the corresponding labels.

**Note_1:** We would now create a new `aug_data` folder outside of the `data` folder. This `aug_data` folder supposed to have the same architechture as the original data folder, where inside it have `train`, `test` and `val` folder, as well as having an `images` and `labels` folder each.

**Note_2:** By running the augmentation **60 times**, the number of training, testing and validation data in `aug_data` should be 60 times more than the data inside `data` folder

## 5. Prepare Labels
- From the labels **json** files, we now create a Tensorflow DataFrame labels for `train`, `test`, and `val` dataset.
- 0 - None, 1 - Face.

## 6. Combine Label and Image Samples
- Now we create a TensorFlow Dataset by zipping the labels and the images together.
- We also do random shufflling, the shuffle should be higher than the number of images inside each dataset.
- We also set the batch to 8, which means we are taking 8 sets of label-image at a time

## 7. Build our deep learning model
- We uses the VGG16 model, since it already have 15 CNN 2D layers, but we modify the last few layers of the model.
- We add 4 additional layers, 2 Classification layers and 2 Regression layers
  - Classification layers uses relu and sigmoid function, output expected to be 1 since it can either be face or no face
  - Regression layers uses relu and sigmoid function, output expected to be 4 represented 2 points each have x and y to build a rectangle
- The optimizer is Adam with addition learning rate decay formula as: $$\frac{\left(\frac{1}{0.75} - 1\right)}{batches\_per\_epoch}$$

- The loss function are different for 2 classification and regression layers:
  - For Regression, we are taking the squared differences between true and predicted width and height, where width is the difference of 2 x values and height is the difference of 2 y values.
  - For Classification, we uses **BinaryCrossEntropy**
  - The total loss is Regression loss + **0.5*** Classification loss

## 8. Plot the performance
- We plot the performance of Total Loss, Classification Loss and Regression Loss as line chart

## 9. Making Prediction
- We apply the test set for prediction
- We visualize our prediction
## 10. Real time Prediction
- Requires OpenCV GUI
- Live face dectection.
