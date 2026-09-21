from __future__ import annotations  # For class method return type hinting

import os
from collections.abc import Sequence
from time import sleep

from dotenv import load_dotenv
from sqlalchemy import (
    TIMESTAMP,
    Column,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
load_dotenv(override=True)

db_username = os.environ.get("MYSQL_USERNAME")
db_password = os.environ.get("MYSQL_PASSWORD")
db_host = os.environ.get("MYSQL_HOST")
db_name = os.environ.get("MYSQL_DATABASE_NAME")
db = create_engine(
    "mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8"
    % (db_username, db_password, db_host, db_name)
)
Session = sessionmaker(db)


class BaseRecord(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True)  # index
    model_name = Column(String(256), index=True)
    task_name = Column(String(256))
    create_time = Column(TIMESTAMP(True), server_default=func.now())
    # NOTE: record_name = model_name + "#" + task_name

    def insert(self, max_retries: int = 2):
        for attempt in range(max_retries):
            try:
                with Session() as session:
                    session.add(self)
                    session.commit()
                break
            except SQLAlchemyError as e:
                if attempt < max_retries - 1:
                    sleep(2**attempt)  # Exponential backoff
                else:
                    raise e

    @classmethod
    def query(cls, **kwargs) -> Sequence[BaseRecord]:
        """Query records by keyword arguments.
        Input:
            **kwargs: keyword arguments
        Return:
            Sequence[BaseRecord]: records
        Example:
            >>> PropertyRecord.query(model_name="TEST_DP_v1", task_name="task1")
        """
        with Session() as session:
            return session.query(cls).filter_by(**kwargs).all()

    @classmethod
    def count(cls, **kwargs) -> int:
        """Count records by keyword arguments.
        Input:
            **kwargs: keyword arguments
        Return:
            int: number of records found
        Example:
            >>> PropertyRecord.count(model_name="TEST_DP_v1", task_name="task1")
        """

        with Session() as session:
            return session.query(cls).filter_by(**kwargs).count()
