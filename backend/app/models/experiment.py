"""
Experiment data model.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base


class Experiment(Base):
    """
    Model for storing experimental test data.
    
    Attributes:
        id: Primary key
        name: Experiment name/identifier
        temperature: Test temperature in Celsius
        humidity: Relative humidity percentage
        cycles: Number of test cycles
        failures: Number of failures observed
        test_duration_hours: Total test duration
        data_file_path: Path to uploaded data file
        created_at: Creation timestamp
    """
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Test conditions
    temperature = Column(Float, nullable=False)  # Celsius
    humidity = Column(Float, nullable=True)  # Percentage
    temperature_cycles = Column(Integer, nullable=True)
    
    # Results
    cycles = Column(Integer, nullable=True)
    failures = Column(Integer, default=0)
    test_duration_hours = Column(Float, nullable=True)
    mean_time_to_failure = Column(Float, nullable=True)
    
    # File reference
    data_file_path = Column(String(500), nullable=True)
    data_format = Column(String(50), nullable=True)  # CSV, XLSX, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name='{self.name}', temp={self.temperature}C)>"
