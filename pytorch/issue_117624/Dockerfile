FROM python:3.10

WORKDIR /app
COPY test_issue_117624.py buggy_code.py requirements.txt ./
RUN pip install -r requirements.txt
CMD ["pytest", "-sx"]