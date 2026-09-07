from sqlalchemy import JSON, Column

from lambench.databases.base_table import BaseRecord


class CalculatorRecord(BaseRecord):
    __tablename__ = "calculator"

    metrics = Column(JSON)
