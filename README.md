# EdgeOptimisation
This project implements the client server architecture for CNN splitting for optimisation on Edge Devices.

Intermediate activations after the split are sent from client to the server for inference by server which sends the results back to the client

## Prerequisites 

* Convolutional Neural Networks
* Python
* Pytorch 
* Numpy 
* Flask
* Matplotlib
## Installation 
```
git clone https://github.com/tanmayj2020/EdgeOptimisation
pip install -r requirements.txt
```

**Server-** 
```
python server.py
```

**Client-** 
```
python client.py
```
REMEMBER : Change the server's IP in the settings.json

### Mode's of client 
1. Prediction with confidence score and latency for an image
2. Latecy Plot for different splits of CNN for an image 