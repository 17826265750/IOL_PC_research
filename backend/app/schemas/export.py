"""
Pydantic schemas for export operations.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"


class ReportConfig(BaseModel):
    """Configuration for report generation."""
    include_charts: bool = Field(
        default=True,
        description="Include charts in the report"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence intervals in the report"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include recommendations in the report"
    )
    language: str = Field(
        default="en",
        description="Report language (en or zh)"
    )
    page_size: str = Field(
        default="A4",
        description="Page size for PDF (A4 or letter)"
    )


class ExportRequest(BaseModel):
    """Request schema for exporting data."""
    format: ExportFormat = Field(
        ...,
        description="Export format"
    )
    prediction_id: Optional[int] = Field(
        None,
        description="Prediction ID to export"
    )
    experiment_id: Optional[int] = Field(
        None,
        description="Experiment ID to export"
    )
    config: Optional[ReportConfig] = Field(
        None,
        description="Report configuration"
    )


class ExportResponse(BaseModel):
    """Response schema for export operations."""
    success: bool = Field(..., description="Export success status")
    message: str = Field(..., description="Status message")
    download_url: Optional[str] = Field(
        None,
        description="URL to download the exported file"
    )
    file_size: Optional[int] = Field(
        None,
        description="File size in bytes"
    )


class PDFExportRequest(BaseModel):
    """Request schema for PDF export."""
    prediction_id: Optional[int] = Field(None, description="Prediction ID")
    experiment_id: Optional[int] = Field(None, description="Experiment ID")
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Direct data to export (alternative to ID)"
    )
    config: Optional[ReportConfig] = Field(
        None,
        description="Report configuration"
    )


class ExcelExportRequest(BaseModel):
    """Request schema for Excel export."""
    prediction_id: Optional[int] = Field(None, description="Prediction ID")
    experiment_id: Optional[int] = Field(None, description="Experiment ID")
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Direct data to export (alternative to ID)"
    )
    include_rainflow: bool = Field(
        default=True,
        description="Include rainflow cycle data if available"
    )
    include_mission_profile: bool = Field(
        default=True,
        description="Include mission profile data if available"
    )


class BatchExportRequest(BaseModel):
    """Request schema for batch exporting multiple items."""
    prediction_ids: List[int] = Field(
        default_factory=list,
        description="List of prediction IDs to export"
    )
    experiment_ids: List[int] = Field(
        default_factory=list,
        description="List of experiment IDs to export"
    )
    format: ExportFormat = Field(
        ...,
        description="Export format"
    )
    config: Optional[ReportConfig] = Field(
        None,
        description="Report configuration"
    )
