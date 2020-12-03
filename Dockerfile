FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

MAINTAINER Dayyan Smith <dayyan.smith@gmail.com>


RUN pip install scikit-learn

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Set up the program in the image
COPY domainbed/ /opt/ml/code/domainbed/
COPY sm_entrypoint.py /opt/ml/code/sm_entrypoint.py
WORKDIR /opt/ml/code/

ENTRYPOINT ["python", "sm_entrypoint.py"]