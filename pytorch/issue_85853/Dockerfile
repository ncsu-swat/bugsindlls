FROM python:3.8 

WORKDIR /app
COPY test_issue_85853.py requirements.txt ./
RUN pip install -r requirements.txt

CMD ["pytest", "-sx"]