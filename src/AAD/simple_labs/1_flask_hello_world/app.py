from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test2")
def test2():
    return "<p>Testing path of /test2!</p>"

if __name__== "__main__":
    app.run(debug=True)