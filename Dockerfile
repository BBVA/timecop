FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN apt-get update && apt-get install -y redis-server python3-celery python-celery-common python3-redis


COPY ./ /app

COPY ./requirements.txt /tmp/
RUN pip3 install numpy sqlalchemy
RUN pip install --upgrade --no-deps statsmodels
RUN pip3 install --requirement /tmp/requirements.txt

RUN pip install -U numpy

RUN pip3 install pandas scipy patsy matplotlib numdifftools seaborn
RUN pip3 install pyflux
RUN pip3 install pyramid-arima
RUN pip3 install tensorflow==1.14.0
RUN pip3 install -U statsmodels
RUN pip3 install tbats
RUN pip3 install celery
RUN pip install redis
RUN service redis-server start
RUN pip install h5py


COPY ./config/timeout.conf /etc/nginx/conf.d/
RUN chmod -R g=u /etc/passwd /app
