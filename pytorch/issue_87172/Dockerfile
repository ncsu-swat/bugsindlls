
FROM python:3.8 

WORKDIR /app
COPY test_issue_87172.py requirements.txt ./
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]