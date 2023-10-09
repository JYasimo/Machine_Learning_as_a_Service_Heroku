# Use a base image with Python 3.9
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install statsmodels library
RUN pip3 install statsmodels

# Copy your FastAPI application code into the container
COPY ./app /app

# Install Python dependencies and libgomp1
RUN apt-get update && apt-get install -y libgomp1 && pip3 install -r requirements.txt


# Install python-multipart package
RUN pip install python-multipart

# Copy your models (if any) into the container
COPY ./models /models

# Copy your dataset (if any) into the container
COPY ./dataset /dataset


# Define the command to run your FastAPI application
CMD sh -c 'uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}'


