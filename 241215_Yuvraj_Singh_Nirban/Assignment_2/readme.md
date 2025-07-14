# Assignment 2
# Setup Instructions

## Install all the dependencies:


```bash
pip install -r requirements.txt
```

## Directory arrangement:
```
Assignment_2/
├── readme.md
├── close.csv
├── stock_data.csv
├── Assignment_2.py
├── Assignment_2.ipynb
└── requirements.txt
```
The dataset can be downloaded from the link -[dataset](https://drive.google.com/drive/folders/1DV20S2S0NwkAjrlNyd4x3J0maUcdbq0E?usp=sharing)

Now you are all set to run the notebook in your own native environment!

## Objective
The objective of the following assignment is to predict the missing values from a real world stock price dataset.
## Data Description
1. Each row represents one trading day.
2. Columns include: Date, Close, Open, High, Low, Volume. 
3. Some of the values are missing in the columns except the “Date” (obviously).
## Approach 
The approach involves use of a time series model LSTM (Long Short Term Memory), below is the brief overview of main highlights of the implementation of the LSTM model. 
### Normalisation and Preprocessing of Data
Normalisation is one of the essential steps in implementation of a LSTM model since it forbids features with a different scale to disproportionately hijack the model. The normalisation can be easily done using either the *Keras Normalization* or *Scikit-Learn's Standard Scaler* (in the current implementation the latter is used since we are not dealing with tensors features here *i.e.* *High,Low,Open,Volume*).

Secondly, there are a few data which have missing values, these values are filled according to the previous value to ensure that the LSTM cannot get stuck. (Its better to fill these values with something instead of just zeros).

### Prior Duration
For prediction of the close for the missing values, we train the model using the estimation that a lot of stock market prices depend on the performance of the past week closes and thus 7 previous days are taken into account in predicting the working close of the current day. Some days which do not have enough prior days in the dataset are predicted with less than 7 days of data (2 days to be exact). 

The model architecture is shown below: 
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">17,664</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘

<span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">59,333</span> (231.77 KB)

<span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">19,777</span> (77.25 KB)

<span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)

<span style="font-weight: bold"> Optimizer params: </span><span style="color: #00af00; text-decoration-color: #00af00">39,556</span> (154.52 KB)
</pre>

### Prediction 
A total of 19 values are predicted in total and saved in a separate *.csv* files in the same directory. 

## Author
### ***NovaPrime2077***


