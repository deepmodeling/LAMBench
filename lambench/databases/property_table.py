from sqlalchemy import Column, Float

from lambench.databases.base_table import BaseRecord


class PropertyRecord(BaseRecord):
    __tablename__ = "property"

    property_rmse = Column(Float)
    property_mae = Column(Float)

    def to_dict(self) -> dict:
        return {
            "property_rmse": self.property_rmse,
            "property_mae": self.property_mae,
        }
