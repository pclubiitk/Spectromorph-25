### Dataset
The dataset contains 540 emoji images organised into 18 different classes.

### Transformation
Apart from the usual transformations, the transformations of random rotation and random horizontal flipping are applied to the training dataset for data augmentation.

### Model Architecture
A simple CNN model is used which includes $3$ convolutional and $2$ dense layers. `ReLU` is used as the activation function in the model, while `Dropout` is used as the regularization technique.

### Training
During training, the model is evaluated using the cross entropy loss criterion, and is optimized using `Adam`.

### Final Results
Training Accuracy= 75.23%
Test Accuracy= 67.59%