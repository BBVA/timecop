import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String,LargeBinary,DATETIME,Text,Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from struct import *
from sqlalchemy.sql import *
import datetime
import pandas as pd

from collections import defaultdict

from sqlalchemy.inspection import inspect




Base = declarative_base()

class Model(Base):
    __tablename__ = 'models'
    TS_name = Column(String(250), nullable=False,primary_key=True)
    TS_winner_name = Column(String(250), nullable=False)
    TS_model = Column(LargeBinary)
    TS_model_params = Column(String(250))
    TS_metric = Column(Numeric)

    TS_update = Column('TS_update', DATETIME, index=False, nullable=False,primary_key=True,default=datetime.datetime.utcnow)


class TS(Base):
    __tablename__ = 'timeseries'
    TS_name = Column(String(250), nullable=False,primary_key=True)
    TS_data = Column(Text())
    TS_update = Column('TS_update', DATETIME, index=False, nullable=False,primary_key=True,default=datetime.datetime.utcnow)


def init_database():

    Base = declarative_base()

    class Model(Base):
        __tablename__ = 'models'
        TS_name = Column(String(250), nullable=False,primary_key=True)
        TS_winner_name = Column(String(250), nullable=False)
        TS_model = Column(LargeBinary())
        TS_model_params = Column(String(250))
        TS_metric = Column(Numeric)
        TS_update = Column('TS_update', DATETIME, index=False, nullable=False,primary_key=True,default=datetime.datetime.utcnow)

    class TS(Base):
        __tablename__ = 'timeseries'
        TS_name = Column(String(250), nullable=False,primary_key=True)
        TS_data = Column(Text())
        TS_update = Column('TS_update', DATETIME, index=False, nullable=False,primary_key=True,default=datetime.datetime.utcnow)


    DB_NAME = 'sqlite:///Timecop_modelsv1.db'
    engine = create_engine(DB_NAME)
    #self.__db.echo = True
    Base.metadata.create_all(engine)


def get_ts(name):

    DB_NAME = 'sqlite:///Timecop_modelsv1.db'
    engine = create_engine(DB_NAME)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    query = session.query(TS)
    salida = session.query(TS).filter(TS.TS_name == name).order_by(desc('TS_update')).first()
    return ( salida.TS_data)


def set_ts(name, data):

    DB_NAME = 'sqlite:///Timecop_modelsv1.db'
    engine = create_engine(DB_NAME)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()


    new_TS = TS(TS_name=name, TS_data= data)
    session.add(new_TS)
    session.commit()


def new_model(name, winner, model,params,metric):

    DB_NAME = 'sqlite:///Timecop_modelsv1.db'
    engine = create_engine(DB_NAME)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    print ("Model saved")
    #print(model)

    new_model = Model(TS_name=name, TS_winner_name = winner, TS_model=bytearray(model),TS_model_params= params,TS_metric=metric)
    session.add(new_model)
    session.commit()

def get_best_model(name):
    DB_NAME = 'sqlite:///Timecop_modelsv1.db'
    engine = create_engine(DB_NAME)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Model)
    # for mt in query.all():
    #     print (unpack('H',(mt.TS_model)))
    #     print (mt.TS_name)
    #     print (mt.TS_winner_name)
    #     print (mt.TS_update)
    #filter(Note.message.like("%somestr%")
    winner_model_type = session.query(Model).filter(Model.TS_name.like('winner%')).order_by(desc('TS_update')).first()

    salida = session.query(Model).filter(Model.TS_name == name).filter(Model.TS_winner_name == winner_model_type.TS_winner_name).order_by(desc('TS_update')).first()
    return ( salida.TS_winner_name,salida.TS_model,salida.TS_model_params)


def query_to_dict(rset):
    result = defaultdict(list)
    for obj in rset:
        instance = inspect(obj)
        for key, x in instance.attrs.items():
            result[key].append(x.value)
    return result


def get_all_models(name):
    DB_NAME = 'sqlite:///Timecop_modelsv1.db'
    engine = create_engine(DB_NAME)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Model).filter(Model.TS_name.like(name)).all()
    df = pd.DataFrame(query_to_dict(query))
    df.drop('TS_model',axis=1,inplace=True)
    return (df[['TS_name', 'TS_winner_name','TS_update']])



def get_winners(name):
    DB_NAME = 'sqlite:///Timecop_modelsv1.db'
    engine = create_engine(DB_NAME)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Model)
    # for mt in query.all():
    #     print (unpack('H',(mt.TS_model)))
    #     print (mt.TS_name)
    #     print (mt.TS_winner_name)
    #     print (mt.TS_update)
    #filter(Note.message.like("%somestr%")
    winner_model_type = session.query(Model).filter(Model.TS_name.like(name)).filter(Model.TS_name.like('winner%')).order_by(desc('TS_update')).all()

    df = pd.DataFrame(query_to_dict(winner_model_type))
    df.drop('TS_model',axis=1,inplace=True)
    return (df[['TS_name', 'TS_winner_name','TS_update']])
