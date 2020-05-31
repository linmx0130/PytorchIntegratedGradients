import flask
from flask import Flask, request, redirect, send_from_directory
from flask import render_template
import string
import random
import os 
import core
import json
app = Flask(__name__)
with open('imgnet1000.json') as f:
    ImageNetLabel = json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

def generateFilename(filename):
    lastname = filename.split('.')[-1]
    firstname = ''.join([random.choice(string.ascii_letters) for i in range(12)])
    return firstname + '.' + lastname

@app.route("/runvis", methods=["POST"])
def runvis():
    imgFile = request.files['image-upload']
    if imgFile.filename == "":
        return redirect(request.url)
    newFilename = generateFilename(imgFile.filename)
    newFilepath = os.path.join('upload', newFilename)
    imgFile.save(newFilename)
    gradImg, cropImg, clsResult = core.integratedGradient(newFilename, sampleSize=60)
    gradImg.save(newFilepath + "grad.jpg")
    cropImg.save(newFilepath + "crop.jpg")

    return render_template('vis.html', cls=ImageNetLabel[str(clsResult)], imgFn = newFilename)

@app.route('/img/<filename>')
def imgRoute(filename):
    return send_from_directory('upload', filename)

if __name__=="__main__":
    app.run("127.0.0.1", 8000)
