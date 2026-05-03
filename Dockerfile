FROM python:3.13-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ports
EXPOSE 8501 8000
# command
CMD ["bash"]