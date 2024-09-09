# Use Python 3.10 slim image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

RUN mkdir -p /app/exports/charts
RUN chmod 777 /app/exports/charts

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and images into the container
COPY app.py .
COPY images ./images

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]