import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

input_image = Image.open("golf ball.jpeg")
preprocess = transforms.Compose([
        transforms.Resize(256) , transforms.CenterCrop(224) ,transforms.ToTensor() , transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

model = models.alexnet(pretrained=True)
model.eval()

output = model(input_batch)

# print("[NORMALISING] the outputs")
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print("---------------------------------------------------------")
# print("[PREDICTING] the labels")
    # print("---------------------------------------------------------")
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())