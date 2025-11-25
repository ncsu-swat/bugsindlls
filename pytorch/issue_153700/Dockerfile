FROM python:3.11

WORKDIR /app
COPY test_issue_153700.py requirements.txt ./
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]