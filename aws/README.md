## Image Identification on AWS
### PROJECT PLAN and PROCEDURE

- Using ResNet-50, a convolutional neural network that is 50 layers deep. We load a pretrained version of the neural network trained on more than a million images from the ImageNet database
- A client will upload images and the system will output the classification/object name and a confidence score for that classifcation.
- We intend to add elasticity on AWS, perhaps using containers if time allows.


### Diagram








### Notes
- Gallery with images from Imagenet (to test our image classification tool) https://github.com/EliSchwartz/imagenet-sample-images/blob/master/gallery.md
- https://github.com/Internet2/class-adv-spr2024-aiml
- https://github.com/asitkdash/Object-Detection-Using-Keras/blob/master/object%20detection%20using%20keras.ipynb
- https://paperswithcode.com/datasets?task=object-detection
- https://cocodataset.org/#home
- https://aws.amazon.com/tutorials/train-deep-learning-model-aws-ec2-containers/
- https://aws.amazon.com/machine-learning/containers/
