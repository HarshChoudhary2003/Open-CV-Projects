"""VisionAI - Report Generation API (PDF + CSV)"""

import csv
import io
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select

from app.core.security import get_current_user
from app.core.database import get_db, DetectionEvent, AlertEvent
from app.core.config import settings

router = APIRouter()


@router.get("/reports/csv/detections")
async def export_detections_csv(
    hours: int = Query(24, ge=1, le=720),
    camera_id: Optional[str] = None,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    since = datetime.utcnow() - timedelta(hours=hours)
    q = select(DetectionEvent).where(DetectionEvent.timestamp >= since)
    if camera_id:
        q = q.where(DetectionEvent.camera_id == camera_id)
    rows = (await db.execute(q)).scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "camera_id", "timestamp", "class", "confidence",
                     "track_id", "x1", "y1", "x2", "y2"])
    for r in rows:
        writer.writerow([r.id, r.camera_id, r.timestamp.isoformat(),
                         r.object_class, r.confidence, r.track_id,
                         r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=detections_{hours}h.csv"},
    )


@router.get("/reports/csv/alerts")
async def export_alerts_csv(
    hours: int = Query(24, ge=1, le=720),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    since = datetime.utcnow() - timedelta(hours=hours)
    q = select(AlertEvent).where(AlertEvent.timestamp >= since)
    rows = (await db.execute(q)).scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "camera_id", "timestamp", "type", "severity",
                     "description", "acknowledged"])
    for r in rows:
        writer.writerow([r.id, r.camera_id, r.timestamp.isoformat(),
                         r.alert_type, r.severity, r.description, r.acknowledged])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=alerts_{hours}h.csv"},
    )


@router.get("/reports/pdf/summary")
async def export_pdf_summary(
    hours: int = Query(24, ge=1, le=720),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Generate a PDF summary report using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return Response(
            content="Install reportlab: pip install reportlab",
            status_code=501,
        )

    since = datetime.utcnow() - timedelta(hours=hours)
    det_rows = (await db.execute(
        select(DetectionEvent).where(DetectionEvent.timestamp >= since)
    )).scalars().all()
    alert_rows = (await db.execute(
        select(AlertEvent).where(AlertEvent.timestamp >= since)
    )).scalars().all()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"VisionAI Platform - Summary Report", styles["Title"]))
    elements.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    elements.append(Paragraph(f"Period: Last {hours} hours", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Detection summary table
    class_counts: dict = {}
    for r in det_rows:
        class_counts[r.object_class] = class_counts.get(r.object_class, 0) + 1

    det_data = [["Object Class", "Count"]] + [[k, v] for k, v in sorted(class_counts.items(), key=lambda x: -x[1])]
    t = Table(det_data, colWidths=[200, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    elements.append(Paragraph("Detection Summary", styles["Heading2"]))
    elements.append(t)
    elements.append(Spacer(1, 20))

    # Alert summary
    elements.append(Paragraph(f"Total Alerts: {len(alert_rows)}", styles["Normal"]))
    critical = sum(1 for r in alert_rows if r.severity == "CRITICAL")
    high = sum(1 for r in alert_rows if r.severity == "HIGH")
    elements.append(Paragraph(f"Critical: {critical}  High: {high}", styles["Normal"]))

    doc.build(elements)
    buf.seek(0)

    return Response(
        content=buf.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=visionai_report_{hours}h.pdf"},
    )
