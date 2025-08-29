from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def default_print():
    return "<p>Default Print message: = Hello, World!</p>"

@app.route("/getname", methods = ['GET', 'POST'])
def get_name():
    data = request.json()
    user_entered_name = str(data['name'])
    print(user_entered_name)
    return jsonify(data)

@app.route("/getmean", methods = ["GET", 'post'])
# can call with http://127.0.0.1:5000/getmean?num1=10&num2=20 too
def get_mean():
    number_1 = int(request.args.get("num1"))
    number_2 = int(request.args.get("num2"))
    return str((number_1 + number_2 )/2)

def python_get_mean(number_1, number_2):
    pass

@app.route("/getmax", methods = ['GET'])
# can call with http://127.0.0.1:5000/getmax?num1=10&num2=20 too
def get_max():
    # Alternative to request.get_json
    # data = request.get_json()
    # number_1 = int(data['num1'])
    # number_2 = int(data['num2'])
    number_1 = int(request.args.get("num1"))
    number_2 = int(request.args.get("num2"))
    return str(max(number_1, number_2))
    
if __name__== "__main__":
    app.run(debug=True)

# RUNNING THE APP: python -m flask --app flask/quickstart.py run
# RUNNING APP OVER WEB: python -m flask --app flask/quickstart.py run --host=0.0.0.0
# GETTING DATA FROM JSON: curl http://127.0.0.1:5000/getname -H "Content-Type:application/json" -d '{"Name": "ajinkya"}'