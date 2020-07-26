# AutoFAQ
AutoFAQ is qa widget for static sites with on-device processing. It takes data from the current page and makes the index. 

## Model

AutoFAQ uses MobileBERT. I planned to fine-tune it and remove half of layers to decrease the model size. Unfortuently, I didn't manage to convert the model from PyTorch to Tensorflow.js. (Please, help me if you could). 

## Demo
python3 -m http.server 8080
