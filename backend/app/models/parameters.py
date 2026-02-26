"""
Model parameters storage.
"""
from sqlalchemy import Column, Integer, String, Float, Text, JSON
from sqlalchemy.orm import relationship
from app.db.database import Base


class ModelParameters(Base):
    """
    Model for storing lifetime prediction model parameters.
    
    Attributes:
        id: Primary key
        name: Parameter set name
        model_type: Type of model (ARRHENIUS, EYRING, COFFIN_MASON, etc.)
        parameters: JSON field storing model-specific parameters
        description: Description of parameter set
    """
    __tablename__ = "model_parameters"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    
    # Arrhenius parameters
    activation_energy = Column(Float, nullable=True)  # eV
    pre_exponential_factor = Column(Float, nullable=True)
    
    # Coffin-Manson parameters  
    fatigue_ductility_coefficient = Column(Float, nullable=True)
    fatigue_ductility_exponent = Column(Float, nullable=True)
    
    # General stress parameters
    stress_exponent = Column(Float, nullable=True)
    boltzmann_constant = Column(Float, default=8.617e-5)  # eV/K
    
    # Additional parameters stored as JSON
    additional_params = Column(JSON, nullable=True)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="parameters")
    
    def __repr__(self) -> str:
        return f"<ModelParameters(id={self.id}, name='{self.name}', type='{self.model_type}')>"
