# # Start from a base image
# FROM python:3.8-slim-buster

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file into the container
# COPY requirements.txt .

# # Install the Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application files into the container
# COPY . .

# # Set the environment variable for the Flask app
# ENV FLASK_APP=app.py

# # Expose the port that the application will listen on
# EXPOSE 5000

# # Run the application when the container starts
# CMD ["python3", "app.py", "--host=0.0.0.0", "--port=5000"]
# # ENTRYPOINT python3 app.py
# # CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]


FROM python:3.8-slim-buster

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 6000

CMD ["python3", "app.py"]