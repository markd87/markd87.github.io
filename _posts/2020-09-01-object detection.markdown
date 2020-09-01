---
layout: post
title: "Object Detection"
date: 2020-09-01 21:50:00 +0100
tags: deeplearning pytorch
---

One of the algorithms for object detection in images is called
Faster R-CNN [paper](https://arxiv.org/abs/1506.01497).

Unlike the typical image classification task, which given an input image returns a class. The task of object detection is much harder, and many clever techniques have been developed to solve this task.

The objective is clear, a network needs to be able to both provide a bounding box (i.e. box coordinates) indicating where it thinks an object belonging to one of the training classes is found, as well as the class it self. Therefore, the network combines both a classification task for the object class as well as a regression task for the bounding box coordinates. A key idea with the bounding boxes, is that they are not regressed from scratch, rather the regression adjusts a number of initial reference boxes (specifically 9 anchor boxes), which is easier.

The common idea is still to use a CNN to obtain feature maps at different scales, i.e. different network depths, as this contain encoded information within each neuron's receptive field (the portion of the original image it "sees"), and the deeper the layer the larger the receptive field and the complexity of the features.

The feature map is used to propose potential regions of object, the proposed region is further fed into two separate fully-connected layers performing the classification and regression tasks.

There are plenty more specific implementation details as can be read in the paper, but that's the idea.

Here I wanted to try out the pre-trained model using PyTorch.

The Google Colab notebook can be found [here](https://colab.research.google.com/drive/1zt--4f1v1o9Mmd481xKfYI4aF8Z9vhxm?usp=sharing).

Imports:

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
import numpy as np

from PIL import Image
import cv2
import matplotlib.pyplot as plt
```

Load pre-trained Faster R-CNN model.
The ResNet50 indicates that the backbone of the network used for the feature map generation is the ResNet50 CNN.

```python
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Set model to evaluation mode (need for dropout and batchnorm which work differently under training and evaluation)
model.eval()
```

The model was originally trained on an object detection dataset called COCO (Common Objects in Context),
which has 90 classes (not including background class, i.e. no object).

```python
# Predefined object classes
COCO_CATEGORIES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
```

The model's output looks as follows for a sample image:

```python
  # Example output
  img  = Image.open('dog_ball.jpg')
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  out = model([img])
  out
```

Output:

```python
[{'boxes': tensor([[646.0035, 449.6353, 905.4247, 735.7059],
          [ 49.0856, 543.5223, 242.2890, 736.4569],
          [164.4843, 577.5748, 204.7126, 619.7700]], grad_fn=<StackBackward>),
  'labels': tensor([18, 37,  1]),
  'scores': tensor([0.9989, 0.9896, 0.2459], grad_fn=<IndexBackward>)}]
```

It is a list of dictionaries (one dictionary per input image).
The output contains three keys: boxes, labels and scores with tensor values.
The boxes contains the identified boxes as a tensor with 4 values per box corresponding to the top left and bottom right corners.
The labels contain the identified classes (one per box).
The scores contain the probability of the classification, which is binary, either the object belongs to the class or not.

The detection function:

```python
def detect(img_path, save_path, threshold=0.9):
  """
  Use Faster R-CNN to detect objects in an image.
  Filter objects with score below threshold
  """

  # Load image
  img  = Image.open(img_path)

  # Convert to tensor
  transform = T.Compose([T.ToTensor()])
  img = transform(img)

  # Input image to model and get output
  out = model([img]) # expects list of images

  # Get output indices correponding to object having a score above threshold, for filtering of the boxes and labels
  indxs = np.where(out[0]['scores'].detach().numpy() >= threshold)[0]

  # Get detected classes labels
  detected_classes = [COCO_CATEGORIES[i] for i in out[0]['labels'].numpy()[indxs]]

  # Get class probabilities
  scores = [out[0]['scores'].detach().numpy()[i] for i in indxs]

  # Get the boxes corresponding to detected objects as numpy arrays
  boxes = [out[0]['boxes'].detach().numpy()[i] for i in indxs]

  # Create rectangles in the format OpenCV expects, two tuples of top left
  # and top right corners
  rects = [[(box[0],box[1]), (box[2], box[3])] for box in boxes]

  # Use OpenCV to draw rectangle and text on image
  img = cv2.imread(img_path)

  # Need to convert from BGR to RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

  # Loop over bounding boxes and class and add to the image
  for rect, dclass, score in zip(rects, detected_classes, scores):

    x, y = rect[0]
    # Draw Rectangle with the coordinates
    cv2.rectangle(img,
                  rect[0],
                  rect[1],
                  color=(255, 0, 0),
                  thickness=3);

    # Add the class label to the image as text
    cv2.putText(img,
                f"{dclass}: {score:.2f}",
                (int(x), int(y)-10),  # shift text vertically from box
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (250,0,0),
                thickness=4);

  # Show image
  plt.imshow(img)

  # Save image
  # OpenCV expects BGR to save properly
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);
  cv2.imwrite(save_path, img)
```

Examples:

![dogball](/assets/objectdetection/dog_ball_detected.jpg)

![marta](/assets/objectdetection/marta_parrot_detected.jpg)

I actually didn't see the bird at the lower left corner before!

![marta](/assets/objectdetection/family_detected.jpg)

Little people are correctly identified.

The next step would be to utilize the pre-trained model (transfer learning) to
detect new classes of objects.
