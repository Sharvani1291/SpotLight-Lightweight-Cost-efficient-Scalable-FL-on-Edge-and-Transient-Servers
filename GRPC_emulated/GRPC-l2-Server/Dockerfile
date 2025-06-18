FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . . 



# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py with the specified arguments when the container launches
EXPOSE 50051

CMD [ "python", "l2-server.py" ]
