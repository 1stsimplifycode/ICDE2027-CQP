FROM python:3.12-slim
WORKDIR /cqp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cqp/ cqp/
COPY src/ src/
CMD ["sh", "-c", "python -m pytest -q cqp/test_cqp.py && python cqp/demo.py"]
