from  flask import Flask,request,jsonify
from classifier import get_prediction

app = Flask(__name__)

@app.route("/")
def homepage() :
    return "welcome to my homepage"

@app.route("/predict-digit", methods = ["POST"])

def predict_data() :
    img = request.files.get("digit")
    p = get_prediction(img)
    return jsonify({
        "prediction of digit" : p
    })
if __name__ == "__main__" :
    app.run(debug = True)