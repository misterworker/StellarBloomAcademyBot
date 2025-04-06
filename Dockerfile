# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port that uvicorn will run on
EXPOSE 8000

# Run uvicorn with the app on start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
