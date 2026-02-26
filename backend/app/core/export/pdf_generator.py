"""
PDF report generator for lifetime prediction reports.

Generates professional engineering reports including:
- Title and timestamp
- Input parameters table
- Model used and equation
- Prediction results with confidence intervals
- Charts (embedded as images)
"""
import os
import logging
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import (
    HexColor, black, white, grey, darkblue, darkgreen,
    red, blue, green
)
from reportlab.lib.styles import (
    getSampleStyleSheet, ParagraphStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image, Flowable
)
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


# Model equation references
MODEL_EQUATIONS = {
    "coffin-manson": r"N_f = A \cdot (\Delta T_j)^{-\alpha}",
    "coffin-manson-arrhenius": r"N_f = A \cdot (\Delta T_j)^{-\alpha} \cdot \exp\left(\frac{E_a}{k \cdot T_j}\right)",
    "norris-landzberg": r"N_f = A \cdot (\Delta T_j)^{-\alpha} \cdot f^{\beta} \cdot \exp\left(\frac{E_a}{k \cdot T_{j,max}}\right)",
    "cips-2008": r"N_f = K \cdot (\Delta T_j)^{\beta_1} \cdot \exp\left(\frac{\beta_2}{T_{j,m}+273}\right) \cdot t_{on}^{\beta_3} \cdot I^{\beta_4} \cdot V^{\beta_5} \cdot D^{\beta_6}",
    "lesit": r"N_f = K \cdot (\Delta T_j)^{-\alpha} \cdot \exp\left(\frac{E_a}{k \cdot T_{j,min}}\right)",
}

MODEL_DESCRIPTIONS = {
    "coffin-manson": "Basic Coffin-Manson model considering only temperature swing amplitude.",
    "coffin-manson-arrhenius": "Coffin-Manson model with Arrhenius term for mean temperature effects.",
    "norris-landzberg": "Norris-Landzberg model including frequency and maximum temperature effects.",
    "cips-2008": "CIPS 2008 (Bayerer) model with comprehensive stress factors including current, voltage, and bond wire diameter.",
    "lesit": "LESIT model focusing on temperature swing and minimum temperature effects.",
}


@dataclass
class ReportConfig:
    """Configuration for PDF report generation."""
    title: str = "IOL PC Board Lifetime Prediction Report"
    author: str = "IOL PC Research System"
    subject: str = "Lifetime Prediction Analysis"
    keywords: str = "lifetime, prediction, reliability, IGBT"
    page_size: str = "A4"  # "A4" or "letter"
    include_charts: bool = True
    include_confidence: bool = True
    language: str = "en"  # "en" or "zh"


class PDFGenerator:
    """
    PDF report generator for lifetime prediction results.

    Generates professional engineering reports with tables, charts,
    and detailed analysis results.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize PDF generator.

        Args:
            config: Report configuration options
        """
        self.config = config or ReportConfig()
        self.page_size = A4 if self.config.page_size == "A4" else letter
        self.width, self.height = self.page_size
        self.margin = 20 * mm

        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_styles()

        # Register fonts for Chinese support
        self._setup_fonts()

    def _setup_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=darkblue,
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=darkblue,
            spaceBefore=12,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))

        # Subsection heading style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=black,
            spaceBefore=8,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))

        # Normal text style
        self.styles.add(ParagraphStyle(
            name='ReportNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            fontName='Helvetica'
        ))

        # Small text style
        self.styles.add(ParagraphStyle(
            name='ReportSmall',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=4,
            fontName='Helvetica'
        ))

        # Table header style
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=white,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))

        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=red,
            fontName='Helvetica-Bold'
        ))

    def _setup_fonts(self):
        """
        Setup fonts for PDF generation including Chinese font support.

        Attempts to register a Chinese font for multilingual support.
        Falls back to standard fonts if Chinese font is not available.
        """
        # Try to register common Chinese fonts
        chinese_fonts = [
            'C:/Windows/Fonts/msyh.ttc',  # Microsoft YaHei
            'C:/Windows/Fonts/simhei.ttf',  # SimHei
            'C:/Windows/Fonts/simsun.ttc',  # SimSun
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux
            '/System/Library/Fonts/PingFang.ttc',  # macOS
        ]

        self.chinese_font_registered = False
        for font_path in chinese_fonts:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('Chinese', font_path))
                    self.styles['ReportTitle'].fontName = 'Chinese'
                    self.styles['SectionHeading'].fontName = 'Chinese'
                    self.styles['SubsectionHeading'].fontName = 'Chinese'
                    self.chinese_font_registered = True
                    logger.info(f"Registered Chinese font: {font_path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to register font {font_path}: {e}")
                    continue

    def generate_lifetime_report(
        self,
        prediction: Dict[str, Any],
        parameters: Dict[str, Any],
        mission_profile: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> bytes:
        """
        Generate a complete lifetime prediction PDF report.

        Args:
            prediction: Prediction result data
            parameters: Model input parameters
            mission_profile: Mission profile data (optional)
            output_path: Optional file path to save the PDF

        Returns:
            PDF content as bytes
        """
        buffer = BytesIO()

        # Create PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
            author=self.config.author,
            title=self.config.title,
            subject=self.config.subject
        )

        # Build story (content elements)
        story = []

        # Title page
        story.extend(self._create_title_page(prediction, parameters))

        # Page break
        story.append(PageBreak())

        # Input parameters section
        story.extend(self._create_parameters_section(parameters))

        # Model description section
        story.extend(self._create_model_section(prediction))

        # Results section
        story.extend(self._create_results_section(prediction))

        # Charts section (if enabled)
        if self.config.include_charts:
            story.append(PageBreak())
            story.extend(self._create_charts_section(prediction, parameters))

        # Recommendations section
        story.extend(self._create_recommendations_section(prediction))

        # Footer with timestamp
        story.append(Spacer(1, 10 * mm))
        story.append(Paragraph(
            f"<i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            self.styles['ReportSmall']
        ))

        # Build PDF
        doc.build(story)

        # Get PDF content
        pdf_content = buffer.getvalue()
        buffer.close()

        # Save to file if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_content)

        return pdf_content

    def _create_title_page(
        self,
        prediction: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> List[Flowable]:
        """Create the title page of the report."""
        elements = []

        # Title
        elements.append(Paragraph(self.config.title, self.styles['ReportTitle']))
        elements.append(Spacer(1, 5 * mm))

        # Subtitle
        prediction_name = prediction.get('name', 'Unnamed Prediction')
        elements.append(Paragraph(
            f"Analysis: {prediction_name}",
            self.styles['SectionHeading']
        ))
        elements.append(Spacer(1, 10 * mm))

        # Summary table
        model_type = prediction.get('model_type', 'Unknown')
        lifetime_years = prediction.get('predicted_lifetime_years')
        lifetime_cycles = prediction.get('predicted_lifetime_cycles')

        summary_data = [
            ['Report Property', 'Value'],
            ['Prediction ID', str(prediction.get('id', 'N/A'))],
            ['Model Type', self._format_model_name(model_type)],
            ['Created', self._format_timestamp(prediction.get('created_at'))],
        ]

        if lifetime_years is not None:
            summary_data.append(['Predicted Lifetime', f"{lifetime_years:.2f} years"])
        if lifetime_cycles is not None:
            summary_data.append(['Cycles to Failure', f"{lifetime_cycles:,.0f}"])

        damage = prediction.get('total_damage')
        if damage is not None:
            summary_data.append(['Total Damage', f"{damage:.4f}"])

        confidence = prediction.get('confidence_level')
        if confidence is not None and self.config.include_confidence:
            summary_data.append(['Confidence Level', f"{confidence:.1%}"])

        summary_table = Table(summary_data, colWidths=[60 * mm, 80 * mm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F5F5F5')]),
        ]))

        elements.append(summary_table)
        elements.append(Spacer(1, 10 * mm))

        return elements

    def _create_parameters_section(self, parameters: Dict[str, Any]) -> List[Flowable]:
        """Create the input parameters section."""
        elements = []

        elements.append(Paragraph("Input Parameters", self.styles['SectionHeading']))
        elements.append(Spacer(1, 5 * mm))

        # Build parameter table
        param_data = [['Parameter', 'Value', 'Unit']]

        # Common thermal parameters
        thermal_params = {
            'delta_Tj': ('Junction Temperature Swing', '°C'),
            'Tj_max': ('Maximum Junction Temperature', '°C'),
            'Tj_mean': ('Mean Junction Temperature', '°C'),
            'Tj_min': ('Minimum Junction Temperature', '°C'),
            'Tc': ('Case Temperature', '°C'),
        }

        # Electrical parameters
        electrical_params = {
            'I': ('Current', 'A'),
            'V': ('Voltage', 'V'),
            'I_RMS': ('RMS Current', 'A'),
            'f': ('Switching Frequency', 'Hz'),
        }

        # Time parameters
        time_params = {
            't_on': ('On Time', 's'),
            't_off': ('Off Time', 's'),
            'period': ('Cycle Period', 's'),
        }

        # Physical parameters
        physical_params = {
            'D': ('Bond Wire Diameter', 'µm'),
            'Rth': ('Thermal Resistance', 'K/W'),
            'Cth': ('Thermal Capacitance', 'J/K'),
        }

        # Environment parameters
        env_params = {
            'Ta': ('Ambient Temperature', '°C'),
            'humidity': ('Humidity', '%'),
        }

        # Combine all parameter mappings
        all_params = {
            **thermal_params,
            **electrical_params,
            **time_params,
            **physical_params,
            **env_params
        }

        # Add parameters to table
        for key, (name, unit) in all_params.items():
            if key in parameters:
                value = parameters[key]
                formatted_value = self._format_value(value)
                param_data.append([name, formatted_value, unit])

        # Add any additional parameters not in the mapping
        for key, value in parameters.items():
            if key not in all_params:
                formatted_value = self._format_value(value)
                param_data.append([key, formatted_value, '-'])

        param_table = Table(param_data, colWidths=[60 * mm, 40 * mm, 20 * mm])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F5F5F5')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(param_table)
        elements.append(Spacer(1, 10 * mm))

        return elements

    def _create_model_section(self, prediction: Dict[str, Any]) -> List[Flowable]:
        """Create the model description section."""
        elements = []

        elements.append(Paragraph("Model Information", self.styles['SectionHeading']))
        elements.append(Spacer(1, 5 * mm))

        model_type = prediction.get('model_type', 'Unknown')
        formatted_name = self._format_model_name(model_type)

        # Model name and description
        model_info = [
            ['Property', 'Value'],
            ['Model', formatted_name],
            ['Description', MODEL_DESCRIPTIONS.get(model_type, 'Custom model')],
        ]

        model_table = Table(model_info, colWidths=[50 * mm, 90 * mm])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F5F5F5')]),
        ]))

        elements.append(model_table)
        elements.append(Spacer(1, 5 * mm))

        # Equation (as text since we can't easily render LaTeX)
        elements.append(Paragraph("Model Equation:", self.styles['SubsectionHeading']))
        equation = MODEL_EQUATIONS.get(model_type, "Custom equation")
        elements.append(Paragraph(f"<i>{equation}</i>", self.styles['ReportNormal']))
        elements.append(Spacer(1, 10 * mm))

        return elements

    def _create_results_section(self, prediction: Dict[str, Any]) -> List[Flowable]:
        """Create the results section."""
        elements = []

        elements.append(Paragraph("Prediction Results", self.styles['SectionHeading']))
        elements.append(Spacer(1, 5 * mm))

        # Main results table
        results_data = [['Result', 'Value', 'Status']]

        lifetime_years = prediction.get('predicted_lifetime_years')
        lifetime_cycles = prediction.get('predicted_lifetime_cycles')
        damage = prediction.get('total_damage')
        confidence = prediction.get('confidence_level')

        # Add lifetime results
        if lifetime_years is not None:
            status, color = self._get_lifetime_status(lifetime_years, 'years')
            results_data.append([
                'Predicted Lifetime',
                f"{lifetime_years:.2f} years",
                status
            ])

        if lifetime_cycles is not None:
            status, _ = self._get_lifetime_status(lifetime_cycles, 'cycles')
            results_data.append([
                'Cycles to Failure',
                f"{lifetime_cycles:,.0f}",
                status
            ])

        # Add damage result
        if damage is not None:
            status, _ = self._get_damage_status(damage)
            results_data.append([
                'Total Damage',
                f"{damage:.4f}",
                status
            ])

        # Add confidence interval
        if confidence is not None and self.config.include_confidence:
            ci_lower = prediction.get('confidence_interval_lower')
            ci_upper = prediction.get('confidence_interval_upper')
            if ci_lower is not None and ci_upper is not None:
                results_data.append([
                    f'{confidence:.0%} Confidence Interval',
                    f"[{ci_lower:.2f}, {ci_upper:.2f}] years",
                    'OK'
                ])

        results_table = Table(results_data, colWidths=[60 * mm, 50 * mm, 30 * mm])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F5F5F5')]),
        ]))

        elements.append(results_table)

        # Add status-colored row backgrounds
        for i, row in enumerate(results_data[1:], start=1):
            if len(row) > 2:
                status = row[2]
                if status == 'CRITICAL':
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), HexColor('#FFCCCC')),
                    ]))
                elif status == 'WARNING':
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), HexColor('#FFF4CC')),
                    ]))
                elif status == 'OK':
                    results_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), HexColor('#CCFFCC')),
                    ]))

        elements.append(Spacer(1, 10 * mm))

        return elements

    def _create_charts_section(
        self,
        prediction: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> List[Flowable]:
        """Create charts section with embedded plots."""
        elements = []

        elements.append(Paragraph("Analysis Charts", self.styles['SectionHeading']))
        elements.append(Spacer(1, 5 * mm))

        # Create lifetime comparison chart
        try:
            img_bytes = self._create_lifetime_chart(prediction, parameters)
            if img_bytes:
                img = Image(img_bytes, width=160 * mm, height=100 * mm)
                elements.append(img)
                elements.append(Spacer(1, 5 * mm))
                elements.append(Paragraph(
                    "Figure 1: Lifetime Prediction Overview",
                    self.styles['ReportSmall']
                ))
                elements.append(Spacer(1, 10 * mm))
        except Exception as e:
            logger.warning(f"Failed to create lifetime chart: {e}")

        # Create damage distribution chart
        try:
            img_bytes = self._create_damage_chart(prediction, parameters)
            if img_bytes:
                img = Image(img_bytes, width=160 * mm, height=80 * mm)
                elements.append(img)
                elements.append(Spacer(1, 5 * mm))
                elements.append(Paragraph(
                    "Figure 2: Damage Accumulation Analysis",
                    self.styles['ReportSmall']
                ))
                elements.append(Spacer(1, 10 * mm))
        except Exception as e:
            logger.warning(f"Failed to create damage chart: {e}")

        return elements

    def _create_recommendations_section(self, prediction: Dict[str, Any]) -> List[Flowable]:
        """Create the recommendations section."""
        elements = []

        elements.append(Paragraph("Recommendations", self.styles['SectionHeading']))
        elements.append(Spacer(1, 5 * mm))

        recommendations = self._generate_recommendations(prediction)

        for i, rec in enumerate(recommendations, 1):
            elements.append(
                Paragraph(f"{i}. {rec}", self.styles['ReportNormal'])
            )

        # Add notes if present
        notes = prediction.get('notes')
        if notes:
            elements.append(Spacer(1, 5 * mm))
            elements.append(Paragraph("Notes:", self.styles['SubsectionHeading']))
            elements.append(Paragraph(notes, self.styles['ReportNormal']))

        elements.append(Spacer(1, 10 * mm))

        return elements

    def _create_lifetime_chart(
        self,
        prediction: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Optional[bytes]:
        """Create lifetime prediction chart."""
        lifetime_years = prediction.get('predicted_lifetime_years')
        if lifetime_years is None:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Lifetime bar chart
        categories = ['Predicted\nLifetime', '5-Year\nTarget', '10-Year\nTarget']
        values = [lifetime_years, 5, 10]
        colors = [blue, green, darkgreen]

        bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Years', fontsize=10)
        ax1.set_title('Lifetime Comparison', fontsize=11, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, max(15, lifetime_years * 1.2))

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)

        # Stress factors pie chart
        stress_factors = {}
        if 'delta_Tj' in parameters:
            stress_factors['Temp Swing'] = parameters['delta_Tj']
        if 'I' in parameters:
            stress_factors['Current'] = parameters['I']
        if 'Tj_max' in parameters:
            stress_factors['Max Temp'] = parameters['Tj_max']

        if stress_factors:
            labels = list(stress_factors.keys())
            sizes = list(stress_factors.values())
            # Normalize for visualization
            total = sum(sizes)
            sizes = [s / total * 100 for s in sizes]

            wedges, texts, autotexts = ax2.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=['#FF9999', '#66B2FF', '#99FF99']
            )
            ax2.set_title('Stress Factor Distribution', fontsize=11, fontweight='bold')

        plt.tight_layout()

        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        return buffer.getvalue()

    def _create_damage_chart(
        self,
        prediction: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Optional[bytes]:
        """Create damage accumulation chart."""
        damage = prediction.get('total_damage')
        lifetime_cycles = prediction.get('predicted_lifetime_cycles')

        if damage is None and lifetime_cycles is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 3))

        if lifetime_cycles is not None:
            # Create damage accumulation curve
            years = np.linspace(0, 20, 100)
            cycles_per_year = lifetime_cycles / (prediction.get('predicted_lifetime_years', 10) or 10)
            total_cycles = lifetime_cycles

            # Damage accumulation (linear approximation)
            damage_accumulation = years / (prediction.get('predicted_lifetime_years', 10) or 10)

            ax.plot(years, damage_accumulation, 'b-', linewidth=2, label='Damage Accumulation')
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Failure Threshold')

            # Mark current point if damage is known
            if damage is not None:
                current_year = damage * (prediction.get('predicted_lifetime_years', 10) or 10)
                ax.plot(current_year, damage, 'ro', markersize=8, label='Current Damage')

            ax.set_xlabel('Time (Years)', fontsize=10)
            ax.set_ylabel('Damage Ratio', fontsize=10)
            ax.set_title('Damage Accumulation Over Time', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            ax.set_ylim(0, 1.2)
        elif damage is not None:
            # Simple damage gauge
            categories = ['Remaining Life', 'Consumed Life']
            values = [max(0, 1 - damage), min(1, damage)]
            colors = [green, red]

            ax.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
                   startangle=90)
            ax.set_title(f'Damage Status: {damage:.1%}', fontsize=11, fontweight='bold')

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close(fig)

        return buffer.getvalue()

    def _generate_recommendations(self, prediction: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on prediction results."""
        recommendations = []

        lifetime_years = prediction.get('predicted_lifetime_years')
        damage = prediction.get('total_damage')
        safety_factor = prediction.get('safety_factor', 1.0)

        if lifetime_years is not None:
            if lifetime_years < 1:
                recommendations.append(
                    "<b>CRITICAL:</b> Predicted lifetime is less than 1 year. "
                    "Immediate design review is strongly recommended."
                )
            elif lifetime_years < 5:
                recommendations.append(
                    "<b>WARNING:</b> Predicted lifetime is less than 5 years. "
                    "Consider design improvements to increase reliability."
                )
            elif lifetime_years < 10:
                recommendations.append(
                    "<b>NOTICE:</b> Predicted lifetime is less than 10 years. "
                    "Monitor field performance closely."
                )
            else:
                recommendations.append(
                    "Predicted lifetime meets typical reliability requirements (10+ years)."
                )

        if damage is not None:
            if damage >= 1.0:
                recommendations.append(
                    "Damage accumulation indicates failure is predicted. "
                    "Review operating conditions."
                )
            elif damage >= 0.8:
                recommendations.append(
                    "Damage accumulation is high (>80%). Design margins are limited."
                )
            elif damage >= 0.5:
                recommendations.append(
                    "Damage accumulation is moderate (>50%). "
                    "Consider preventive maintenance scheduling."
                )

        if safety_factor < 1.0:
            recommendations.append(
                "Safety factor is less than 1.0 - design is not conservative."
            )

        if not recommendations:
            recommendations.append("No specific recommendations at this time.")

        return recommendations

    def _format_model_name(self, model_type: str) -> str:
        """Format model type for display."""
        names = {
            "coffin-manson": "Coffin-Manson",
            "coffin-manson-arrhenius": "Coffin-Manson-Arrhenius",
            "norris-landzberg": "Norris-Landzberg",
            "cips-2008": "CIPS 2008 (Bayerer)",
            "lesit": "LESIT",
        }
        return names.get(model_type, model_type.replace('-', ' ').title())

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.4g}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)

    def _format_timestamp(self, timestamp: Any) -> str:
        """Format a timestamp for display."""
        if timestamp is None:
            return "N/A"
        if isinstance(timestamp, str):
            return timestamp[:19]  # Truncate microseconds
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')

    def _get_lifetime_status(self, value: float, units: str) -> Tuple[str, str]:
        """Get status text and color for lifetime value."""
        if units == 'years':
            if value < 1:
                return "CRITICAL", "red"
            elif value < 5:
                return "WARNING", "orange"
            elif value < 10:
                return "NOTICE", "blue"
            else:
                return "OK", "green"
        else:  # cycles
            if value < 10000:
                return "LOW", "orange"
            else:
                return "OK", "green"

    def _get_damage_status(self, damage: float) -> Tuple[str, str]:
        """Get status text and color for damage value."""
        if damage >= 1.0:
            return "FAILURE", "red"
        elif damage >= 0.8:
            return "HIGH", "orange"
        elif damage >= 0.5:
            return "MODERATE", "yellow"
        else:
                return "LOW", "green"


def generate_prediction_report(
    prediction: Dict[str, Any],
    parameters: Dict[str, Any],
    mission_profile: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    config: Optional[ReportConfig] = None
) -> bytes:
    """
    Convenience function to generate a PDF report.

    Args:
        prediction: Prediction result data
        parameters: Model input parameters
        mission_profile: Mission profile data (optional)
        output_path: Optional file path to save the PDF
        config: Optional report configuration

    Returns:
        PDF content as bytes
    """
    generator = PDFGenerator(config)
    return generator.generate_lifetime_report(
        prediction=prediction,
        parameters=parameters,
        mission_profile=mission_profile,
        output_path=output_path
    )
