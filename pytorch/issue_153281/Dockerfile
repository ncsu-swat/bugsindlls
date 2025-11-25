FROM python:3.11

WORKDIR /app
COPY test_issue_153281.py requirements.txt ./
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]