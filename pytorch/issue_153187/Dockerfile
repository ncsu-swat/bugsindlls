FROM python:3.12.3

WORKDIR /app
COPY test_issue_153187.py requirements.txt ./
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]