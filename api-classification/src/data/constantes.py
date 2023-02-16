import os

BASE_DIR = os.getcwd()
SERVICE_ACCOUNT = os.environ.get('SERVICE_ACCOUNT')
SERVICE_KEY = os.path.join(BASE_DIR,'.private-key.json')