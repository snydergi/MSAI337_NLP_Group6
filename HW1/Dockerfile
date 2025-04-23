FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ADD starter.py .
ADD wiki2.test.txt . 
ADD wiki2.train.txt .
ADD wiki2.valid.txt .
RUN pip install transformers
CMD ["python", "./starter.py"]
