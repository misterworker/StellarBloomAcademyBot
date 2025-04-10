# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the PYTHONPATH environment variable so Python can resolve src imports
ENV PYTHONPATH=/app/src

# Expose the port uvicorn will listen on
EXPOSE 8000

# Run uvicorn using the src layout
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
