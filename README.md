# Online Inception Learning

This is an example of how pretrained convolutional neural network for image recognition can learn like 
humans from what it sees in real time.

It uses pretrained Inception v3 network. The key trick is to remove the last layer of the network
which returns predicted categories and replace it with the categories of your choise.

In such case the network has the ability to "see" but don't know what it is seeing, so the user give
it a hint by "naming" the things in front of the camera. Surprisingly the network needs ~10-30 sec of learning to 
distinguish out of 2 or more categories.


## Prerequisites
* Install reequirements:

```pip install -r requirements.txt```

* Connect camera to your computer

* Download:

```curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz```

* extract this file project's:

```data/inception_dir```

## Run:

```python train_online.py```

* to quit hit the `q` on your keyboard

## How it works

* The network has 2 states:
  * inference - predicting what it sees but not learning
  * learning
  
* After run the network is in the inference state
  * you will see the image from your camera with the inscription on the top panel:
  
     ```CATEGORY: [NOTHING] ACCURACY: -1```
     
   * It means that nothing has been presented to the network yet, so it can't make any predictions
   
## Training

### Single training session
During the training session you always need to train (meaning show) the network all categories of your choise

1. prepare categories which you want to train the network and show one of it to the camera
  * eg. CATEGORY_1: you in the glasses; CATEGORY_2: you without the glasses
  * eg. CATEGORY_1: apple; CATEGORY_2: banana; CATEGORY_3: orange

2. press and hold a single key from 1 to 9 for ~5 sec.
  * it will start the training state
  * the training will last as long as you press the key
  * when you relies the key it will switch to the inference state
  
3. you can move slowly your category object to show it from different angles

3. relies the key

4. now you will see that the network is constantly predicting the category of your choise (from 1 to 9)
with various accuracy level

5. repeat steps 1-3 for different category/categories one by one

### Remarks

* repeat Single training session steps from 2 to 5 times. After that you should see that 
the network is predicting well most or all of your categories. Eventually it should predict all categories without many errors

* If you use 3 or more categories you need to always "remind" the network about previous categories
so bear in mind that on each training session you need to show to the network all categories of your choise,
otherwise it will rapidly forget about previous once.

* Currently it is not possible to save the results of training.