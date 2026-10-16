ARG BASE
FROM ${BASE}
COPY worker.py /opt/worker.py
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "/opt/worker.py"]
