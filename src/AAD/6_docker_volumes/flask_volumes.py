from flask import Flask, request

app = Flask(__name__)

# EndPoint with @ decorator
@app.route("/")
def hello_world():
    return  f'Hello World: Docker Volumes Lab. (available endpoint /ls to see local content)'

@app.route("/ls")
def list_files():
    file_content = open("/app/local_directory/message.txt").read()
    return file_content

if __name__== "__main__":
    # With app.run, you can directly run this as a python script
    app.run(host="0.0.0.0", port=5003)

# Running the app
# RUNNING THE APP: python -m flask --app flask/lab1_quickstart.py run
# RUNNING APP OVER WEB: python -m flask --app flask/lab1_quickstart.py run --host=0.0.0.0
# GETTING DATA FROM JSON: curl http://127.0.0.1:5000/getname -H "Content-Type:application/json" -d '{"Name": "ajinkya"}'