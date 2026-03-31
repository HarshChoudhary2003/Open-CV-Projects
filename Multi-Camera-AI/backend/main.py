from fastapi import FastAPI, HTTPException
from database.db import db
from datetime import datetime

app = FastAPI(title="Multi-Camera AI API", version="1.0")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Multi-Camera ReID Tracking Backend Running"}

@app.get("/events")
def get_events(limit: int = 50):
    return {"events": db.get_recent_events(limit)}

@app.get("/persons/active")
def get_active_persons():
    return {"active_persons": db.list_active_persons()}

@app.post("/alerts")
def create_alert(person_id: str, camera_id: str, reason: str):
    db.add_alert(person_id, camera_id, reason)
    return {"status": "success", "message": f"Alert registered for {person_id}"}
