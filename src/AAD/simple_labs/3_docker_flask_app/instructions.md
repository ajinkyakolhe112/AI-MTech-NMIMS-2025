# Lab 3: Containerizing a Flask App with Docker

**Goal:** Package a Flask application into a Docker container.

### Prerequisites

- Docker is installed and running.
- Completion of Lab 1.

### Files

1.  **`app.py`**: A simple Flask app that will run inside our container.
2.  **`requirements.txt`**: Lists the Python dependency (Flask).
3.  **`Dockerfile`**: The instruction manual for Docker to build our image.

### Understanding the `Dockerfile`

```dockerfile
# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local code to the container
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
```

### Steps

1.  **Build the Docker Image:**
    Open a terminal in this directory (`flask_labs/lab3`) and run the build command. This tells Docker to build an image using the `Dockerfile` in the current directory and tag it with the name `flask-container-lab`.
    ```bash
    docker build . -t flask-container-lab
    ```

2.  **Run the Docker Container:**
    Execute the run command. This starts a container from the image we just built.
    - `-d`: Runs the container in detached mode (in the background).
    - `-p 5000:5000`: Maps port 5000 on your local machine to port 5000 inside the container.
    ```bash
    docker run -d -p 5000:5000 flask-container-lab
    ```

3.  **View the Result:**
    Open a web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000). You should see your Flask app running, served from the Docker container.

4.  **Stopping the Container (Optional):**
    To find and stop the container, run:
    ```bash
    docker ps # Find the CONTAINER ID
    docker stop <CONTAINER ID>
    ```
