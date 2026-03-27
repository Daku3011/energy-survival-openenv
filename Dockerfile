# Use Python 3.10 for OpenEnv compatibility
FROM python:3.10-slim

# Set up a new user 'user' with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies if any (none needed for this env)
# RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=user . /app

# OpenEnv must listen on port 7860 for Hugging Face Spaces
# We run the server using uvicorn directly to ensure the port is bound correctly
CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
