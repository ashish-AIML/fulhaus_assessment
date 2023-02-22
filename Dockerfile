# Start from a base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application files into the container
COPY app.py .
COPY resnet.h5 .

# Expose the port that the application will listen on
EXPOSE 5000

# Run the application when the container starts
CMD ["python", "app.py"]
