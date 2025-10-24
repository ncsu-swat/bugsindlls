FROM python:3.9

WORKDIR /app
COPY test_issue_107942.py requirements.txt ./
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]