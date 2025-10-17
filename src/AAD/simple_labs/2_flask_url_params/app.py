from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def default_print():
    return "<p>Default Print message: = Hello, World!</p>"

@app.route("/getname", methods = ['POST'])
def get_name():
    data = request.get_json()
    return jsonify(data)

@app.route("/getmean", methods = ['GET'])
def get_mean():
    number_1 = int(request.args.get("num1"))
    number_2 = int(request.args.get("num2"))
    return str((number_1 + number_2 )/2)

if __name__== "__main__":
    app.run(debug=True)
