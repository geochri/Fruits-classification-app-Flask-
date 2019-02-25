
"""
Created on Thu Aug 30 22:16:19 2018

@author: Wee Tee
"""

from flask import Flask, render_template, request
import os
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

uploads = os.path.join("static", "uploads")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = uploads

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route("/upload", methods=["POST"])
def upload_file():
    invlabels_dict = {0:'apple', 1:'orange', 2:'pear'}
    resnet18 = models.resnet18(pretrained=True)
    for i, param in resnet18.named_parameters():
        param.requires_grad = False
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 3)
    checkpoint = torch.load("./savemodel/model_best.pth.tar")
    resnet18.load_state_dict(checkpoint['state_dict'])
    resnet18.eval()
    transform_test = transforms.Compose([transforms.ToTensor(),\
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                     std=[0.229, 0.224, 0.225])])    
    imagename = request.files["image"]
    f = os.path.join(app.config["UPLOAD_FOLDER"], imagename.filename)
    imagename.save(f)
    img = Image.open(imagename)
    img = img.resize(size=(224,224))
    img = np.array(img)
    img = transform_test(img)
    output = resnet18(img.reshape(1,3,224,224))
    _, predicted = torch.max(output.data, 1)
    predicted_class = invlabels_dict[predicted.item()]

    return render_template("index.html", user_image = f, output=predicted_class)

if __name__ == "__main__":
   app.run(use_reloader=False)