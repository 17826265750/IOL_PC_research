"""
Prediction record model.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base


class Prediction(Base):
    """
    Model for storing lifetime prediction results.
    
    Attributes:
        id: Primary key
        name: Prediction name/identifier
        model_type: Type of lifetime model used
        predicted_lifetime_years: Predicted lifetime in years
        predicted_lifetime_cycles: Predicted lifetime in cycles
        total_damage: Calculated total damage
        created_at: Creation timestamp
        updated_at: Last update timestamp
        parameters_id: Foreign key to model parameters
        mission_profile_id: Foreign key to mission profile
    """
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False, default="CIPS_ARRHENIUS")
    predicted_lifetime_years = Column(Float, nullable=True)
    predicted_lifetime_cycles = Column(Integer, nullable=True)
    total_damage = Column(Float, nullable=True)
    confidence_level = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    
    parameters_id = Column(Integer, ForeignKey("model_parameters.id"), nullable=True)
    mission_profile_id = Column(Integer, ForeignKey("mission_profiles.id"), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    parameters = relationship("ModelParameters", back_populates="predictions")
    mission_profile = relationship("MissionProfile", back_populates="predictions")
    
    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, name='{self.name}', model_type='{self.model_type}')>"
