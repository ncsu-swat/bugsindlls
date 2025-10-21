FROM python:3.10

WORKDIR /app
COPY test_issue_108021.py requirements.txt ./
RUN pip install -r requirements.txt
CMD ["pytest", "-sx"]