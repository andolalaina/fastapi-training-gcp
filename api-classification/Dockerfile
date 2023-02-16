FROM python:3.10.5

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ARG SERVICE_ACCOUNT="your-service-account@mail.com"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]