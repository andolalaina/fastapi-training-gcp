FROM continuumio/miniconda3

RUN conda create -n env python=3.6.6

RUN echo "source activate env" > ~/.bashrc

ENV PATH /opt/conda/envs/env/bin:$PATH

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

RUN conda install GDAL -y

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
