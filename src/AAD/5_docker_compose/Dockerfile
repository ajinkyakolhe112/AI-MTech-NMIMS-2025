FROM python:3.12

COPY . /app
WORKDIR /app
# Install the application dependencies
RUN pip install -r requirements.txt

EXPOSE 5002

CMD ["python", "flask_app.py"]