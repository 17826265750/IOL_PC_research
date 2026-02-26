"""
Mission profile model for storing temperature and stress profiles.
"""
from sqlalchemy import Column, Integer, String, Float, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class MissionProfile(Base):
    """
    Model for storing mission profile data.
    
    A mission profile defines the environmental and operational conditions
    experienced by the IOL over its lifetime.
    
    Attributes:
        id: Primary key
        name: Profile name
        description: Profile description
        profile_data: JSON field containing temperature/time or stress/time data points
        duration_years: Expected mission duration in years
        cycles_per_year: Number of thermal cycles per year
    """
    __tablename__ = "mission_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    profile_type = Column(String(50), nullable=False, default="THERMAL")
    
    # Profile data as JSON: [{"time": 0, "temperature": 25}, {"time": 3600, "temperature": 85}, ...]
    profile_data = Column(JSON, nullable=False)
    
    # Mission parameters
    duration_years = Column(Float, nullable=False, default=20)
    cycles_per_year = Column(Float, nullable=True)
    duty_cycle = Column(Float, nullable=True)
    
    # Environmental conditions
    min_temperature = Column(Float, nullable=False)
    max_temperature = Column(Float, nullable=False)
    mean_temperature = Column(Float, nullable=True)
    temperature_ramp_rate = Column(Float, nullable=True)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="mission_profile")
    
    def __repr__(self) -> str:
        return f"<MissionProfile(id={self.id}, name='{self.name}', type='{self.profile_type}')>"
