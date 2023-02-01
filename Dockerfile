# Dockerfile
FROM python:3.7
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install h5py tqdm matplotlib pandas pyqtgraph scipy PyQt5
CMD ["python", "./main.py"]
