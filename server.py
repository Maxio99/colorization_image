import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'coloriztaion/colorizers')
sys.path.insert(2, 'colorizaiton/demo_release')
import os
#import torch
#import argparse
import matplotlib.pyplot as plt
from flask import Flask, send_file,request
from colorizers import *
from demo_release import load_img, preprocess_img, postprocess_tens

app = Flask(__name__)

# Load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

@app.route('/', methods=['GET','POST'])
def colorize():
    try:
        # Set the image path
        #image_path = r'C:\Users\gaith\Desktop\colorization\imgs\happy_dog.jpg'
        image_path = request.files['image']
        print(f"request: ", {image_path})
    
    
        # Load and preprocess the image
        img = load_img(image_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    
        # Colorize the image
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs))
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs))
    
        # Save the colorized image to the desktop
        # desktop = os.path.expanduser("~/Desktop")
        # output_path = os.path.join(desktop, "colorized_image.png")
        # plt.imsave(output_path, out_img_siggraph17)
    
        # Send the colorized image as a response
    
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "failed"


if __name__ == '__main__':
    app.run()
