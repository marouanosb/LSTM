FROM python:3.12
ADD model.py .
ADD requirements.txt .
RUN pip install -r requirements.txt
CMD [“python”, “./model.py”] 