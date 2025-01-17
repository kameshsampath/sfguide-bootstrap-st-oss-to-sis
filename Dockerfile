ARG BASE_IMAGE=python:3.11-slim-buster
FROM $BASE_IMAGE
WORKDIR /app

COPY sis/streamlit_app.py /app/streamlit_app.py
COPY entrypoint.sh /app/entrypoint

RUN pip install --upgrade pip && \
    pip install --user 'streamlit==1.35.0' 'snowflake-snowpark-python' 'scikit-learn==1.3.0' 'pandas==2.0.3' 'numpy==1.24.3' 

ENV SERVICE_HOST=0.0.0.0
ENV SERVER_PORT=8080

ENV PATH="/root/.local/bin:${PATH}"
EXPOSE 8080

ENTRYPOINT [ "/app/entrypoint" ]