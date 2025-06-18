FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install grpcio grpcio-tools 

WORKDIR /app

COPY . .

#EXPOSE 8081
ENTRYPOINT [ "python","client.py" ]