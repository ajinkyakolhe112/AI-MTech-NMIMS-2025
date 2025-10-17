from flask import Flask, request
import datetime
import os

app = Flask(__name__)
LOG_FILE = "/logs/timestamps.txt" # This path corresponds to the mounted volume

@app.route("/")
def show_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    return "<p>No logs yet. Visit /log to create some!</p>"

@app.route("/log")
def log_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp}\n")
    return f"<p>Logged: {timestamp}</p><p><a href=\"/">View logs</a></p>"

if __name__== "__main__":
    app.run(host='0.0.0.0', port=5000)

