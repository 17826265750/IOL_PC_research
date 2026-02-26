"""
CRUD operations for database models.
"""
from typing import TypeVar, Generic, Type, List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base CRUD operations with default methods."""
    
    def __init__(self, model: Type[ModelType]):
        """
        Initialize CRUD object.
        
        Args:
            model: SQLAlchemy model class
        """
        self.model = model
    
    def get(self, db: Session, id: int) -> Optional[ModelType]:
        """
        Get a single record by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Model instance or None
        """
        return db.get(self.model, id)
    
    def get_multi(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[ModelType]:
        """
        Get multiple records with pagination.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of model instances
        """
        stmt = select(self.model).offset(skip).limit(limit)
        result = db.execute(stmt)
        return list(result.scalars().all())
    
    def create(
        self, 
        db: Session, 
        obj_in: CreateSchemaType
    ) -> ModelType:
        """
        Create a new record.
        
        Args:
            db: Database session
            obj_in: Pydantic schema with creation data
            
        Returns:
            Created model instance
        """
        obj_in_data = obj_in.model_dump() if hasattr(obj_in, 'model_dump') else obj_in.dict()
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update(
        self,
        db: Session,
        db_obj: ModelType,
        obj_in: UpdateSchemaType | Dict[str, Any]
    ) -> ModelType:
        """
        Update an existing record.
        
        Args:
            db: Database session
            db_obj: Existing model instance
            obj_in: Pydantic schema or dict with update data
            
        Returns:
            Updated model instance
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True) if hasattr(obj_in, 'model_dump') else obj_in.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def delete(self, db: Session, id: int) -> Optional[ModelType]:
        """
        Delete a record by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Deleted model instance or None
        """
        obj = db.get(self.model, id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj
    
    def count(self, db: Session) -> int:
        """
        Count total records.
        
        Args:
            db: Database session
            
        Returns:
            Number of records
        """
        stmt = select(self.model)
        result = db.execute(stmt)
        return len(result.scalars().all())
