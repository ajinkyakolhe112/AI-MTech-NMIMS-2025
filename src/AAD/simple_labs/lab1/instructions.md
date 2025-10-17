# Lab 1: Your First Flask Application

**Goal:** Create and run a simple "Hello, World" web server using Flask.

### Prerequisites

- Python is installed.
- Flask is installed (`pip install -r requirements.txt`).

### Steps

1.  **Review the Code:**
    Open `app.py`. This script imports Flask, creates a web server instance, and defines a single route `/` that returns "Hello, World!".

2.  **Install Dependencies:**
    In your terminal, run the following command in this directory (`flask_labs/lab1`):
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    You can run the app in multiple ways.

    *   **Method 1: Direct Execution (Simple)**
        Because the script has an `if __name__ == "__main__":` block, you can run it directly:
        ```bash
        python app.py
        ```

    *   **Method 2: Using the `flask` Command**
        You can use the Flask command-line tool. This requires setting an environment variable to tell Flask where your app is.
        ```bash
        # For Linux/macOS
        export FLASK_APP=app.py
        flask run

        # For Windows
        set FLASK_APP=app.py
        flask run
        ```

    *   **Method 3: Making the App Accessible on Your Network**
        To have the app accessible by other devices on the same network, use the `--host=0.0.0.0` argument.
        ```bash
        # Using the flask command
        export FLASK_APP=app.py
        flask run --host=0.0.0.0

        # Or by modifying app.py to run on 0.0.0.0 and then using python app.py
        ```

4.  **View the Result:**
    Open a web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000). If you used `--host=0.0.0.0`, you can also access it from another device on your network using your computer's local IP address.
