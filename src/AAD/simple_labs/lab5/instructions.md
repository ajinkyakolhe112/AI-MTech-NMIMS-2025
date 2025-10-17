# Lab 5: Introduction to Docker Compose

**Goal:** Learn to use Docker Compose to define and run a simple multi-container application.

### Prerequisites

- Completion of Labs 3 & 4.
- Docker Compose is installed (it is included with Docker Desktop).

### Understanding the Project Structure

- **`docker-compose.yaml`**: The master file that tells Docker Compose how to build and run our services.
- **`app/`**: A directory containing our Flask web application.
- **`runner/`**: A directory for a second, simple service that just runs a shell script.

### Understanding `docker-compose.yaml`

This file defines two services:

1.  **`web`**: Our Flask application.
    - `build: ./app` tells Compose to build an image from the `Dockerfile` inside the `app` directory.
    - `ports: - "5005:5000"` maps port 5005 on the host to port 5000 in the container.
2.  **`script-runner`**: A simple service.
    - `build: ./runner` builds from the `Dockerfile` in the `runner` directory.

### Steps

1.  **Review the Files:**
    Look inside the `app` and `runner` directories to see their respective `Dockerfile`s and source code. Note that they are simple and self-contained.

2.  **Build and Run with a Single Command:**
    The magic of Docker Compose is running everything with one command. From this directory (`flask_labs/lab5`), run:
    ```bash
    docker-compose up --build
    ```
    - `up`: Starts the services.
    - `--build`: Forces Docker to build the images before starting.

3.  **View the Output:**
    - In your terminal, you will see the logs from both the `web` and `script-runner` services, color-coded for readability.
    - In a web browser, navigate to [http://localhost:5005](http://localhost:5005) to see the Flask app running.

4.  **Stopping the Services:**
    To stop all the running services, press `Ctrl+C` in the terminal where `docker-compose` is running. To clean up completely (remove containers and networks), run:
    ```bash
    docker-compose down
    ```
