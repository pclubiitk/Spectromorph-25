# Assignment 1
The following model tries to use the CNNs(Convolutional Neural Networks) to classify hand-drawn emojis correctly. The notebook file can be run quite easily on a native machine. The following things must be performed before you run the notebook in you machine. 

## Setup Instructions

### 1. Install all the dependencies:
```bash pip install -r requirements.txt
```

### 2. Directory arrangement:
```
project/
├── dataset/
│   ├── beaming-face/
│   ├── cloud/
│   ├── face-spiral/
│   ├── flushed-face/
│   ├── grimacing-face/
│   ├── grinning-face/
│   ├── grinning-squinting/
│   ├── heart/
│   ├── pouting-face/
│   ├── raised-eyebrow/
│   ├── relieved-face/
│   ├── savoring-food/
│   ├── smiling-heart/
│   ├── smiling-horns/
│   ├── smiling-sunglasses/
│   ├── smiling-tear/
│   ├── smirking-face/
│   └── tears-of-joy/
├── project.ipynb
└── requirements.txt
```
The dataset can be downloaded from the link -[dataset](https://drive.google.com/drive/folders/1Uo5WCK3z35z8k4k3gVfHn_N-OoE9rLNt?usp=sharing)

Now you are all set to run the notebook on your own native environment!

## Result of performance metrics
The model was analyzed through some of the evaluation metrics to see if the generalization is better and following are the results:
1. Accuracy and Loss on training sets and validation sets
  <div align="center">
  <img src="https://github.com/NovaPrime2077/Spectromorph-25/blob/main/Assignment_1/Loss.png" alt="Example Image" width="800"/>
  </div>
3. Confusion matrix of the test set: 

<div align="center">
  <img src="https://github.com/NovaPrime2077/Spectromorph-25/blob/main/Assignment_1/CF_matrix.png" alt="Example Image" width="400"/>
</div>

#### Author 
***Yuvraj Singh Nirban*** aka ***NovaPrime2077***
