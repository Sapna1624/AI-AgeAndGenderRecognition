**Objective :**
To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam, using Deep Learning on the Adience dataset.

**About the Project :**
In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.

**Dataset :**
For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it here. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.

**Additional Python Libraries Required :**
**OpenCV**
   pip install opencv-python
   
   
**The contents of this Project :**
opencv_face_detector.pbtxt
opencv_face_detector_uint8.pb
age_deploy.prototxt
age_net.caffemodel
gender_deploy.prototxt
gender_net.caffemodel
a few pictures to try the project on
detect.py
For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

**Steps to follow:**

**1. Face detection with Haar cascade :-  This is a part most of us at least have heard of.**
OpenCV provide direct methods to import Haar cascades and use them to detect face.

**2 . Gender Recognition with CNN :-**
Gender recognition using OpenCV's fisherfaces implementation is quite popular and some of you may have tried or read about it also. But, in this example, I will be using a different approach to recognize gender. This method was introduced by two Israel researchers, Gil Levi and Tal Hassner in 2015. I have used the CNN models trained by them in this example. We are going to use the OpenCV’s dnn package which stands for “Deep Neural Networks”.
In the dnn package, OpenCV has provided a class called Net which can be used to populate a neural network. Furthermore, these packages support importing neural network models from well known deep learning frameworks like caffe, tensorflow and torch. The researchers I had mentioned above have published their CNN models as caffe models. Therefore, we will be using the CaffeImporter import that model into our application.

**3. Age Recognition with CNN :-**
CNN algorithm is used for age recognition. This is almost similar to the gender detection part except that the corresponding prototxt file and the caffe model file are “deploy_agenet.prototxt” and “age_net.caffemodel”.
 Furthermore, the CNN’s output layer (probability layer) in this CNN consists of 8 values for 8 age classes -
(“0–2”, “4–6”, “8–13”, “15–20”, “25–32”, “38–43”, “48–53” and “60-”)

**A caffe model has 2 associated files :-**
1 .prototxt — The definition of CNN goes in here. This file defines the layers in the neural network, each layer’s inputs, outputs and functionality.
2 .caffemodel — This contains the information of the trained neural network (trained model).

**Usage :**
Download my Repository
Open your Command Prompt or Terminal and change directory to the folder where all the files are present.


**Detecting Gender and Age of face in Image Use Command :**
  python detect.py --image <image_name>
Note: The Image should be present in same folder where all the files are present

**Detecting Gender and Age of face through webcam Use Command :**
  python detect.py
Press Ctrl + C to stop the program execution.


**Working:**
Watch the video


**Examples :**

NOTE:- The images have been downloaded from Google.

Once source code has been generated, software must be tested to uncover as many errors as  possible before delivery. It is very important to work the system successfully and achieve high  quality of software. Testing include designing a series of test cases that have a high likelihood of  finding errors by applying software-testing techniques.
System testing makes logical assumptions that if all the parts of the system are correct, the goal  will be successfully achieved. The system should be checked logically. Validations and cross  checks should be there. Avoid duplications of record that cause redundancy of data.
In other Words, Testing is the process of evaluating a system or its component(s) with the intent to find whether it satisfies the specified requirements or not. It is executing a system in order to identify any gaps, errors, or missing requirements in contrary to the actual requirements.

>python detect.py --image girl1.jpg
Gender: Female
Age: 25-32 years


>python detect.py --image boy1.jpg
Gender: Male
Age: 4-6 years


>python detect.py --image girl2.jpg
Gender: Female
Age: 38-43 years    


>python detect.py --image boy2.jpg
Gender: Male
Age: 25-32 years  


>python detect.py --image girl3.jpg
Gender: Female
Age: 4-6 years


>python detect.py --image boy3.jpg
Gender: Male
Age: 60-100 years


>python detect.py --image group1.jpg
Gender: Female, Male, Male
Age: 25-32, 0-2, 8-12 years
