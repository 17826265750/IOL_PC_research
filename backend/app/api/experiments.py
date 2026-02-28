"""
Experiment data CRUD endpoints.

功率模块寿命分析软件 - 实验数据管理API
Author: GSH

Provides endpoints for:
- Creating experiment records
- Reading experiment data
- Updating experiment records
- Deleting experiments
- Listing all experiments
"""
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import csv
import json
from io import StringIO

from app.db.database import get_db
from app.db.crud import CRUDBase
from app.models.experiment import Experiment
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])

# CRUD instance
experiment_crud = CRUDBase(Experiment)


@router.get("", response_model=List[ExperimentResponse])
async def get_experiments(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all experiments with pagination."""
    try:
        experiments = experiment_crud.get_multi(db, skip=skip, limit=limit)
        return experiments
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch experiments: {str(e)}"
        )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Get a specific experiment by ID."""
    experiment = experiment_crud.get(db, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    return experiment


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    request: ExperimentCreate,
    db: Session = Depends(get_db)
):
    """Create a new experiment record."""
    try:
        experiment = experiment_crud.create(db, request)
        return experiment
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {str(e)}"
        )


@router.put("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: int,
    request: ExperimentUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing experiment."""
    experiment = experiment_crud.get(db, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )

    try:
        updated = experiment_crud.update(db, experiment, request)
        return updated
    except Exception as e:
        logger.error(f"Error updating experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update experiment: {str(e)}"
        )


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Delete an experiment."""
    experiment = experiment_crud.get(db, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )

    experiment_crud.delete(db, experiment_id)
    return None


@router.get("/stats/summary")
async def get_experiment_summary(db: Session = Depends(get_db)):
    """Get summary statistics of all experiments."""
    try:
        from sqlalchemy import func, select

        stmt = select(
            func.count(Experiment.id).label("total"),
            func.sum(Experiment.failures).label("total_failures"),
            func.avg(Experiment.temperature).label("avg_temperature"),
            func.avg(Experiment.test_duration_hours).label("avg_duration")
        )
        result = db.execute(stmt).one()

        total_experiments = result.total or 0
        total_failures = result.total_failures or 0
        avg_temperature = float(result.avg_temperature) if result.avg_temperature else 0.0
        avg_duration = float(result.avg_duration) if result.avg_duration else 0.0

        # Calculate MTTF
        if total_failures > 0:
            stmt_duration = select(func.sum(Experiment.test_duration_hours))
            total_duration = db.execute(stmt_duration).scalar() or 0
            mttf = total_duration / total_failures
        else:
            mttf = None

        return {
            "total_experiments": total_experiments,
            "total_failures": total_failures,
            "average_temperature": avg_temperature,
            "average_test_duration": avg_duration,
            "mean_time_to_failure": mttf
        }

    except Exception as e:
        logger.error(f"Error getting experiment summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get summary: {str(e)}"
        )


@router.get("/by/temperature/{min_temp}/{max_temp}", response_model=List[ExperimentResponse])
async def get_experiments_by_temperature_range(
    min_temp: float,
    max_temp: float,
    db: Session = Depends(get_db)
):
    """Get experiments within a temperature range."""
    try:
        from sqlalchemy import select

        stmt = select(Experiment).where(
            Experiment.temperature >= min_temp,
            Experiment.temperature <= max_temp
        )
        result = db.execute(stmt)
        experiments = list(result.scalars().all())

        return experiments

    except Exception as e:
        logger.error(f"Error fetching experiments by temperature: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch experiments: {str(e)}"
        )


@router.post("/import/csv")
async def import_experiments_from_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Import experiments from CSV file.

    Expected CSV columns:
    - name (required)
    - description (optional)
    - temperature (required)
    - humidity (optional)
    - temperature_cycles (optional)
    - cycles (optional)
    - failures (optional, default 0)
    - test_duration_hours (optional)
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are supported"
            )

        # Read CSV content
        content = await file.read()
        csv_text = content.decode('utf-8')

        # Parse CSV
        reader = csv.DictReader(StringIO(csv_text))

        imported = []
        errors = []

        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            try:
                # Validate required fields
                if not row.get('name'):
                    errors.append(f"Row {row_num}: Missing 'name' field")
                    continue

                if not row.get('temperature'):
                    errors.append(f"Row {row_num}: Missing 'temperature' field")
                    continue

                # Create experiment
                experiment_data = ExperimentCreate(
                    name=row['name'],
                    description=row.get('description'),
                    temperature=float(row['temperature']),
                    humidity=float(row['humidity']) if row.get('humidity') else None,
                    temperature_cycles=int(row['temperature_cycles']) if row.get('temperature_cycles') else None,
                    cycles=int(row['cycles']) if row.get('cycles') else None,
                    failures=int(row.get('failures', 0)),
                    test_duration_hours=float(row['test_duration_hours']) if row.get('test_duration_hours') else None
                )

                experiment = experiment_crud.create(db, experiment_data)
                imported.append(experiment)

            except ValueError as e:
                errors.append(f"Row {row_num}: Invalid data - {str(e)}")
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")

        return {
            "imported_count": len(imported),
            "errors": errors,
            "imported": [exp.id for exp in imported]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV import failed: {str(e)}"
        )


@router.get("/{experiment_id}/export")
async def export_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """Export experiment data as JSON."""
    experiment = experiment_crud.get(db, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )

    return {
        "id": experiment.id,
        "name": experiment.name,
        "description": experiment.description,
        "temperature": experiment.temperature,
        "humidity": experiment.humidity,
        "temperature_cycles": experiment.temperature_cycles,
        "cycles": experiment.cycles,
        "failures": experiment.failures,
        "test_duration_hours": experiment.test_duration_hours,
        "mean_time_to_failure": experiment.mean_time_to_failure,
        "data_file_path": experiment.data_file_path,
        "data_format": experiment.data_format,
        "created_at": experiment.created_at.isoformat(),
        "updated_at": experiment.updated_at.isoformat()
    }


@router.post("/{experiment_id}/calculate-mttf")
async def calculate_experiment_mttf(experiment_id: int, db: Session = Depends(get_db)):
    """Calculate and update MTTF for an experiment."""
    experiment = experiment_crud.get(db, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )

    try:
        if experiment.failures > 0 and experiment.test_duration_hours:
            mttf = experiment.test_duration_hours / experiment.failures
        else:
            mttf = None

        # Update experiment
        updated = experiment_crud.update(
            db,
            experiment,
            {"mean_time_to_failure": mttf}
        )

        return {
            "experiment_id": experiment_id,
            "failures": experiment.failures,
            "test_duration_hours": experiment.test_duration_hours,
            "calculated_mttf": mttf
        }

    except Exception as e:
        logger.error(f"Error calculating MTTF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MTTF calculation failed: {str(e)}"
        )


@router.get("/stats/by-temperature")
async def get_stats_by_temperature(db: Session = Depends(get_db)):
    """Get experiment statistics grouped by temperature."""
    try:
        from sqlalchemy import select, func

        stmt = select(
            Experiment.temperature,
            func.count(Experiment.id).label("count"),
            func.sum(Experiment.failures).label("total_failures"),
            func.avg(Experiment.test_duration_hours).label("avg_duration")
        ).group_by(Experiment.temperature).order_by(Experiment.temperature)

        results = db.execute(stmt).all()

        stats = []
        for row in results:
            stats.append({
                "temperature": float(row.temperature),
                "experiment_count": row.count,
                "total_failures": row.total_failures or 0,
                "average_duration": float(row.avg_duration) if row.avg_duration else 0.0
            })

        return {"statistics": stats}

    except Exception as e:
        logger.error(f"Error getting temperature statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics calculation failed: {str(e)}"
        )
