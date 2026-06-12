# Use the official Python 3.10 image from the Docker Hub
FROM python:3.10

# Set the working directory
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
