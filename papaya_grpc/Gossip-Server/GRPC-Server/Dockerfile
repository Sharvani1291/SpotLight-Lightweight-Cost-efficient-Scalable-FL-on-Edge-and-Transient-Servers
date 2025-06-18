FROM  python:3.9.21-slim

WORKDIR /app

COPY . .

RUN pip install grpcio grpcio-tools numpy

#EXPOSE 8088

CMD [ "python3","emulated_client.py" ]
