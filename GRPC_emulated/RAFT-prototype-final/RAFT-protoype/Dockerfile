FROM python:3.11 

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt 
# Ensure the Layers directory is in PYTHONPATH
ENV PYTHONPATH="/app/Layers"
EXPOSE 5000
EXPOSE 50051
EXPOSE 50052

CMD ["python", "raft.py"]
