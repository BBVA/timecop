FROM tiangolo/uwsgi-nginx-flask:python3.6

COPY ./ /app

COPY ./requirements.txt /tmp/
RUN pip3 install numpy sqlalchemy
RUN pip install --upgrade --no-deps statsmodels
RUN pip3 install --requirement /tmp/requirements.txt

RUN pip install -U numpy

RUN pip3 install pandas scipy patsy matplotlib numdifftools seaborn
RUN pip3 install pyflux
RUN pip3 install pyramid-arima
RUN pip3 install tensorflow
RUN pip3 install -U statsmodels

COPY ./config/timeout.conf /etc/nginx/conf.d/
