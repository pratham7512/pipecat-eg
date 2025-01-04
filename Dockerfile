# Dockerfile
FROM python:3.10-bullseye

# Create app directory
WORKDIR /app

# Copy only the requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE ${PORT:-10000}

# Command to run the application
CMD ["python3", "app.py"]