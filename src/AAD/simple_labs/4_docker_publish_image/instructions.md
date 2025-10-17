# Lab 4: Publishing a Docker Image to Docker Hub

**Goal:** Learn how to build a Docker image and publish it to a public registry (Docker Hub) so it can be shared and used anywhere.

### Prerequisites

- Completion of Lab 3.
- A Docker Hub account. If you don't have one, sign up at [https://hub.docker.com/](https://hub.docker.com/).

### Steps

1.  **Log In to Docker Hub via CLI**
    Before you can push an image, you must authenticate with Docker Hub. Use the `docker login` command. When prompted for a password, it is recommended to use a **Personal Access Token** instead of your account password.

    - You can create an access token from your Docker Hub account settings: `Account Settings > Security > New Access Token`.
    ```bash
    docker login -u <YourDockerHubUsername>
    # Paste your Personal Access Token when prompted for a password
    ```

2.  **Build and Tag the Image Correctly**
    To push an image to Docker Hub, you must tag it with your username in the format `<YourDockerHubUsername>/<ImageName>:<Tag>`.

    Replace `<YourDockerHubUsername>` with your actual Docker Hub username in the command below.
    ```bash
    docker build . -t <YourDockerHubUsername>/flask-published-app:latest
    ```

3.  **Push the Image to Docker Hub**
    Now, push the tagged image to the registry.
    ```bash
    docker push <YourDockerHubUsername>/flask-published-app:latest
    ```

4.  **Verify on Docker Hub**
    Go to your Docker Hub profile in a web browser (e.g., `https://hub.docker.com/u/<YourDockerHubUsername>`). You should see your new `flask-published-app` repository.

5.  **Test the Published Image**
    To prove it works, you can pull the image from Docker Hub and run it, just as anyone else would.
    ```bash
    # (Optional) First, remove the local image to ensure you're pulling from the web
    docker rmi <YourDockerHubUsername>/flask-published-app:latest

    # Pull the image from the registry
    docker pull <YourDockerHubUsername>/flask-published-app:latest

    # Run the container from the pulled image
    docker run -d -p 5000:5000 <YourDockerHubUsername>/flask-published-app:latest
    ```
    You can now access the application in your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000), and it will be running from the image you published.
