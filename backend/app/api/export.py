"""
Export endpoints for generating reports and data exports.

Provides endpoints for:
- PDF report generation with charts and tables
- Excel data export with multiple sheets
- CSV data export
- JSON data export
"""
from fastapi import APIRouter, HTTPException, status, Response, Depends
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Dict, Any, Optional
import logging
import json
import csv
from io import StringIO, BytesIO
from datetime import datetime

from app.schemas.prediction import PredictionResponse
from app.schemas.experiment import ExperimentResponse
from app.schemas.rainflow import RainflowResponse, CycleCount
from app.schemas.export import (
    PDFExportRequest,
    ExcelExportRequest,
    ReportConfig,
    ExportResponse
)
from app.core.export.pdf_generator import PDFGenerator, ReportConfig as PDFReportConfig
from app.core.export.excel_generator import ExcelGenerator, ExcelConfig


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["export"])


# ==================== CSV Export Endpoints ====================

@router.post("/csv/predictions")
async def export_predictions_csv(predictions: List[PredictionResponse]) -> Response:
    """Export predictions to CSV format."""
    try:
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "ID", "Name", "Model Type", "Lifetime (years)",
            "Lifetime (cycles)", "Total Damage", "Confidence",
            "Created At", "Notes"
        ])

        # Data rows
        for p in predictions:
            writer.writerow([
                p.id,
                p.name,
                p.model_type,
                p.predicted_lifetime_years or "",
                p.predicted_lifetime_cycles or "",
                p.total_damage or "",
                p.confidence_level or "",
                p.created_at.isoformat() if p.created_at else "",
                p.notes or ""
            ])

        csv_content = output.getvalue()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )

    except Exception as e:
        logger.error(f"Error exporting predictions CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV export failed: {str(e)}"
        )


@router.post("/csv/experiments")
async def export_experiments_csv(experiments: List[ExperimentResponse]) -> Response:
    """Export experiments to CSV format."""
    try:
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "ID", "Name", "Description", "Temperature (C)",
            "Humidity (%)", "Temperature Cycles", "Cycles",
            "Failures", "Duration (hours)", "MTTF (hours)",
            "Created At"
        ])

        # Data rows
        for exp in experiments:
            writer.writerow([
                exp.id,
                exp.name,
                exp.description or "",
                exp.temperature,
                exp.humidity or "",
                exp.temperature_cycles or "",
                exp.cycles or "",
                exp.failures,
                exp.test_duration_hours or "",
                exp.mean_time_to_failure or "",
                exp.created_at.isoformat() if exp.created_at else ""
            ])

        csv_content = output.getvalue()

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )

    except Exception as e:
        logger.error(f"Error exporting experiments CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV export failed: {str(e)}"
        )


@router.post("/csv/rainflow")
async def export_rainflow_csv(
    cycles: List[CycleCount],
    filename: Optional[str] = None
) -> Response:
    """Export rainflow cycle counts to CSV format."""
    try:
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Stress Range", "Mean Value", "Cycles"])

        # Data rows
        for cycle in cycles:
            writer.writerow([
                cycle.stress_range,
                cycle.mean_value,
                cycle.cycles
            ])

        csv_content = output.getvalue()

        fname = filename or f"rainflow_cycles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={fname}"
            }
        )

    except Exception as e:
        logger.error(f"Error exporting rainflow CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV export failed: {str(e)}"
        )


# ==================== JSON Export Endpoints ====================

@router.post("/json/data")
async def export_data_json(
    data: Dict[str, Any],
    filename: Optional[str] = None
) -> Response:
    """Export any data to JSON format."""
    try:
        json_content = json.dumps(data, indent=2, default=str)

        fname = filename or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={fname}"
            }
        )

    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"JSON export failed: {str(e)}"
        )


@router.post("/json/prediction-report")
async def export_prediction_report(
    prediction: PredictionResponse,
    additional_data: Optional[Dict[str, Any]] = None
) -> Response:
    """Export prediction as a formatted JSON report."""
    try:
        report = {
            "report_type": "Lifetime Prediction",
            "generated_at": datetime.now().isoformat(),
            "prediction": {
                "id": prediction.id,
                "name": prediction.name,
                "model_type": prediction.model_type,
                "predicted_lifetime_years": prediction.predicted_lifetime_years,
                "predicted_lifetime_cycles": prediction.predicted_lifetime_cycles,
                "total_damage": prediction.total_damage,
                "confidence_level": prediction.confidence_level,
            },
            "timestamps": {
                "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
                "updated_at": prediction.updated_at.isoformat() if prediction.updated_at else None,
            },
            "notes": prediction.notes
        }

        if additional_data:
            report["additional_data"] = additional_data

        json_content = json.dumps(report, indent=2, default=str)

        fname = f"prediction_report_{prediction.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        return Response(
            content=json_content,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={fname}"
            }
        )

    except Exception as e:
        logger.error(f"Error exporting prediction report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report export failed: {str(e)}"
        )


# ==================== Excel Export Endpoints ====================

@router.post("/excel/data")
async def export_data_excel(
    data: Dict[str, Any],
    sheet_name: str = "Data",
    filename: Optional[str] = None
) -> Response:
    """
    Export data to Excel format.

    Note: This is a simplified CSV-based export.
    For full Excel support with formatting, consider using openpyxl.
    """
    try:
        # Convert to CSV for compatibility
        output = StringIO()

        if isinstance(data, dict):
            # Flatten dictionary
            writer = csv.writer(output)
            writer.writerow(["Key", "Value"])
            for key, value in data.items():
                writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])

        csv_content = output.getvalue()

        fname = filename or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={fname}"
            }
        )

    except Exception as e:
        logger.error(f"Error exporting Excel: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Excel export failed: {str(e)}"
        )


# ==================== PDF Report Generation Endpoints ====================

@router.post("/report/pdf")
async def export_pdf_report(request: PDFExportRequest) -> Response:
    """
    Generate a PDF report for lifetime prediction results.

    Generates a professional PDF report including:
    - Title and timestamp
    - Input parameters table
    - Model description and equation
    - Prediction results with confidence intervals
    - Embedded charts (if enabled)
    - Recommendations

    Request body should contain either:
    - prediction_id: ID of the prediction to export
    - experiment_id: ID of the experiment to export
    - data: Direct data to export (prediction + parameters)
    """
    try:
        # Get data from request
        prediction_data = request.data or {}
        parameters = prediction_data.get('parameters', {})
        mission_profile = prediction_data.get('mission_profile')

        # Create PDF config
        pdf_config = PDFReportConfig(
            include_charts=request.config.include_charts if request.config else True,
            include_confidence=request.config.include_confidence if request.config else True,
            language=request.config.language if request.config else "en",
            page_size=request.config.page_size if request.config else "A4"
        )

        # Generate PDF
        generator = PDFGenerator(config=pdf_config)
        pdf_content = generator.generate_lifetime_report(
            prediction=prediction_data.get('prediction', prediction_data),
            parameters=parameters,
            mission_profile=mission_profile
        )

        # Generate filename
        prediction_name = prediction_data.get('name', prediction_data.get('prediction', {}).get('name', 'report'))
        safe_name = "".join(c for c in prediction_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"lifetime_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_content))
            }
        )

    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {str(e)}"
        )


@router.get("/export/prediction/{prediction_id}/pdf")
async def export_prediction_pdf(prediction_id: int) -> Response:
    """
    Export a prediction as a PDF report by ID.

    Note: This endpoint requires database access to fetch prediction data.
    For now, it returns a template PDF that should be replaced with actual data.
    """
    try:
        # Create a sample report (in real implementation, fetch from DB)
        sample_data = {
            "id": prediction_id,
            "name": f"Prediction {prediction_id}",
            "model_type": "cips-2008",
            "predicted_lifetime_years": 15.5,
            "predicted_lifetime_cycles": 150000,
            "total_damage": 0.45,
            "confidence_level": 0.95,
            "created_at": datetime.now()
        }

        sample_parameters = {
            "delta_Tj": 80,
            "Tj_max": 398,
            "Tj_mean": 350,
            "t_on": 1.0,
            "I": 100,
            "V": 1200,
            "D": 300
        }

        generator = PDFGenerator()
        pdf_content = generator.generate_lifetime_report(
            prediction=sample_data,
            parameters=sample_parameters
        )

        filename = f"prediction_{prediction_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_content))
            }
        )

    except Exception as e:
        logger.error(f"Error exporting prediction PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF export failed: {str(e)}"
        )


@router.get("/export/experiment/{experiment_id}/pdf")
async def export_experiment_pdf(experiment_id: int) -> Response:
    """
    Export an experiment report as PDF by ID.

    Note: This endpoint requires database access to fetch experiment data.
    """
    try:
        # Create a sample experiment report
        sample_data = {
            "id": experiment_id,
            "name": f"Experiment {experiment_id}",
            "model_type": "arrhenius",
            "predicted_lifetime_years": 8.2,
            "predicted_lifetime_cycles": 80000,
            "total_damage": 0.65,
            "confidence_level": 0.90,
            "created_at": datetime.now()
        }

        sample_parameters = {
            "temperature": 150,
            "humidity": 85,
            "cycles": 1000,
            "duration": 500
        }

        generator = PDFGenerator()
        pdf_content = generator.generate_lifetime_report(
            prediction=sample_data,
            parameters=sample_parameters
        )

        filename = f"experiment_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_content))
            }
        )

    except Exception as e:
        logger.error(f"Error exporting experiment PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF export failed: {str(e)}"
        )


# ==================== Excel Export Endpoints (Full openpyxl) ====================

@router.post("/report/excel")
async def export_excel_report(request: ExcelExportRequest) -> Response:
    """
    Generate an Excel report for lifetime prediction results.

    Generates a multi-sheet Excel workbook including:
    - Summary sheet with key results
    - Parameters sheet with all inputs
    - Results sheet with detailed outputs
    - Rainflow cycles sheet (if available)
    - Mission profile sheet (if available)
    - Charts sheet (if enabled)

    Request body should contain either:
    - prediction_id: ID of the prediction to export
    - experiment_id: ID of the experiment to export
    - data: Direct data to export (prediction + parameters)
    """
    try:
        # Get data from request
        prediction_data = request.data or {}
        parameters = prediction_data.get('parameters', {})
        mission_profile = prediction_data.get('mission_profile')
        rainflow_cycles = prediction_data.get('rainflow_cycles')

        # Create Excel config
        excel_config = ExcelConfig(
            include_charts=True,
            include_formulas=False,
            freeze_headers=True
        )

        # Generate Excel
        generator = ExcelGenerator(config=excel_config)
        excel_content = generator.generate_lifetime_report(
            prediction=prediction_data.get('prediction', prediction_data),
            parameters=parameters,
            mission_profile=mission_profile,
            rainflow_cycles=rainflow_cycles
        )

        # Generate filename
        prediction_name = prediction_data.get('name', prediction_data.get('prediction', {}).get('name', 'report'))
        safe_name = "".join(c for c in prediction_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"lifetime_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        return Response(
            content=excel_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(excel_content))
            }
        )

    except Exception as e:
        logger.error(f"Error generating Excel report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Excel generation failed: {str(e)}"
        )


@router.get("/export/prediction/{prediction_id}/excel")
async def export_prediction_excel(prediction_id: int) -> Response:
    """
    Export a prediction as an Excel report by ID.

    Note: This endpoint requires database access to fetch prediction data.
    """
    try:
        # Create a sample report
        sample_data = {
            "id": prediction_id,
            "name": f"Prediction {prediction_id}",
            "model_type": "cips-2008",
            "predicted_lifetime_years": 15.5,
            "predicted_lifetime_cycles": 150000,
            "total_damage": 0.45,
            "confidence_level": 0.95,
            "created_at": datetime.now()
        }

        sample_parameters = {
            "delta_Tj": 80,
            "Tj_max": 398,
            "Tj_mean": 350,
            "t_on": 1.0,
            "I": 100,
            "V": 1200,
            "D": 300
        }

        # Sample rainflow cycles
        sample_cycles = [
            {"stress_range": 100, "mean_value": 50, "cycles": 1000, "damage": 0.1},
            {"stress_range": 80, "mean_value": 45, "cycles": 2500, "damage": 0.15},
            {"stress_range": 60, "mean_value": 40, "cycles": 5000, "damage": 0.2},
        ]

        generator = ExcelGenerator()
        excel_content = generator.generate_lifetime_report(
            prediction=sample_data,
            parameters=sample_parameters,
            rainflow_cycles=sample_cycles
        )

        filename = f"prediction_{prediction_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        return Response(
            content=excel_content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(excel_content))
            }
        )

    except Exception as e:
        logger.error(f"Error exporting prediction Excel: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Excel export failed: {str(e)}"
        )


# ==================== Report Templates ====================

@router.post("/report/lifetime-prediction")
async def generate_lifetime_report(
    prediction: Dict[str, Any],
    parameters: Dict[str, Any],
    mission_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive lifetime prediction report.

    Returns a structured report with all relevant information
    that can be formatted for PDF or other output.
    """
    try:
        report = {
            "report_title": "IOL PC Board Lifetime Prediction Report",
            "generated_at": datetime.now().isoformat(),
            "prediction": prediction,
            "model_parameters": parameters,
            "mission_profile": mission_profile or {},
            "summary": {
                "predicted_lifetime_years": prediction.get("predicted_lifetime_years"),
                "predicted_lifetime_cycles": prediction.get("predicted_lifetime_cycles"),
                "model_used": prediction.get("model_used", "Unknown"),
                "safety_factor": prediction.get("safety_factor", 1.0)
            },
            "recommendations": _generate_recommendations(prediction)
        }

        return report

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )


@router.post("/report/experiment-summary")
async def generate_experiment_summary_report(
    experiments: List[ExperimentResponse],
    summary_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate experiment summary report."""
    try:
        report = {
            "report_title": "Experiment Test Summary Report",
            "generated_at": datetime.now().isoformat(),
            "summary_statistics": summary_stats,
            "experiments": [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "temperature": exp.temperature,
                    "failures": exp.failures,
                    "duration_hours": exp.test_duration_hours,
                    "mttf": exp.mean_time_to_failure
                }
                for exp in experiments
            ]
        }

        return report

    except Exception as e:
        logger.error(f"Error generating experiment report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )


# ==================== Helper Functions ====================

def _generate_recommendations(prediction: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on prediction results."""
    recommendations = []

    lifetime_years = prediction.get("predicted_lifetime_years")
    damage = prediction.get("total_damage")

    if lifetime_years is not None:
        if lifetime_years < 1:
            recommendations.append("CRITICAL: Predicted lifetime is less than 1 year. Immediate design review recommended.")
        elif lifetime_years < 5:
            recommendations.append("WARNING: Predicted lifetime is less than 5 years. Consider design improvements.")
        elif lifetime_years < 10:
            recommendations.append("NOTICE: Predicted lifetime is less than 10 years. Monitor field performance.")
        else:
            recommendations.append("Predicted lifetime meets typical requirements (10+ years).")

    if damage is not None:
        if damage >= 1.0:
            recommendations.append("Damage accumulation indicates failure predicted.")
        elif damage >= 0.8:
            recommendations.append("Damage accumulation is high (>80%). Design margins are limited.")

    safety_factor = prediction.get("safety_factor", 1.0)
    if safety_factor < 1.0:
        recommendations.append("Safety factor is less than 1.0 - this is not conservative.")

    return recommendations


# ==================== Batch Export ====================

@router.post("/batch/export-multiple")
async def export_multiple_formats(
    data: Dict[str, Any],
    formats: List[str] = ["json", "csv"]
) -> Dict[str, str]:
    """
    Export data in multiple formats.

    Returns download URLs or base64 encoded content for each format.
    """
    try:
        results = {}

        if "json" in formats:
            json_content = json.dumps(data, indent=2, default=str)
            results["json"] = json_content

        if "csv" in formats:
            output = StringIO()
            writer = csv.writer(output)

            if isinstance(data, dict):
                for key, value in data.items():
                    writer.writerow([key, value])
            elif isinstance(data, list) and len(data) > 0:
                # Use keys from first item as header
                headers = list(data[0].keys())
                writer.writerow(headers)
                for item in data:
                    writer.writerow([item.get(h) for h in headers])

            results["csv"] = output.getvalue()

        return results

    except Exception as e:
        logger.error(f"Error in batch export: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch export failed: {str(e)}"
        )


@router.get("/templates/{template_name}")
async def get_export_template(template_name: str):
    """Get export template for specific data type."""
    templates = {
        "prediction": {
            "fields": ["name", "model_type", "parameters", "mission_profile"],
            "required": ["name", "model_type"],
            "example": {
                "name": "Sample Prediction",
                "model_type": "cips-2008",
                "parameters": {
                    "delta_Tj": 80,
                    "Tj_max": 398,
                    "t_on": 1.0,
                    "I": 100,
                    "V": 1200,
                    "D": 300
                }
            }
        },
        "experiment": {
            "fields": ["name", "temperature", "cycles", "failures", "test_duration_hours"],
            "required": ["name", "temperature"],
            "example": {
                "name": "High Temperature Test",
                "temperature": 150,
                "cycles": 1000,
                "failures": 5,
                "test_duration_hours": 500
            }
        },
        "rainflow": {
            "fields": ["time", "value"],
            "required": ["time", "value"],
            "example": [
                {"time": 0, "value": 25},
                {"time": 1, "value": 100},
                {"time": 2, "value": 25}
            ]
        }
    }

    if template_name not in templates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_name}' not found. Available: {list(templates.keys())}"
        )

    return templates[template_name]
