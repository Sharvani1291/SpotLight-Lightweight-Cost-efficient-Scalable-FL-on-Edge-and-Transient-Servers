from celery import Celery
from pymongo import MongoClient
from bson import ObjectId
import logging

logging.basicConfig(level=logging.INFO)

app = Celery('tasks', broker='redis://172.22.85.213:6380/0', backend='redis://172.22.85.213:6380/0')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_concurrency=4
)
