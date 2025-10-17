# Lab 2: URL Parameters and Data Handling in Flask

**Goal:** Learn to handle URL parameters and process incoming data in Flask.

### Prerequisites

- Completion of Lab 1.

### Steps

1.  **Review the Code (`app.py`):**
    This application demonstrates two main concepts:
    - **Reading JSON Data:** The `/getname` route accepts `POST` requests with a JSON body (e.g., `{'name': 'YourName'}`) and returns it.
    - **URL Query Parameters:** The `/getmean` route accepts `GET` requests with query parameters (e.g., `/getmean?num1=10&num2=20`) and calculates the mean.

2.  **Install Dependencies:**
    If you haven't already, install Flask:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    Just like in Lab 1, you can run the app in multiple ways.

    *   **Method 1: Direct Execution**
        ```bash
        python app.py
        ```

    *   **Method 2: Using the `flask` Command**
        ```bash
        # For Linux/macOS
        export FLASK_APP=app.py
        flask run

        # For Windows
        set FLASK_APP=app.py
        flask run
        ```

4.  **Test the Endpoints:**

    *   **Testing `/getmean` with a Browser:**
        Open a browser and go to the following URL. The browser should display `15.0`.
        [http://127.0.0.1:5000/getmean?num1=10&num2=20](http://127.0.0.1:5000/getmean?num1=10&num2=20)

    *   **Testing `/getname` with `curl`:**
        To test the `/getname` endpoint, you need to send a `POST` request with a JSON payload. The `curl` command is perfect for this.
        ```bash
        curl http://127.0.0.1:5000/getname -X POST -H "Content-Type: application/json" -d '{"name": "World"}'
        ```
        The terminal will return the JSON you sent: `{"name":"World"}`.
