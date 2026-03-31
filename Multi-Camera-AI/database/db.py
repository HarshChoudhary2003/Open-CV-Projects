import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="events.sqlite"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Tracking events
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                location TEXT
            )
        ''')
        # Alerts (Blacklist hits, etc.)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                reason TEXT NOT NULL
            )
        ''')
        self.conn.commit()

    def add_tracking_event(self, person_id, camera_id, location):
        self.cursor.execute('''
            INSERT INTO tracking_events (person_id, camera_id, location, timestamp) 
            VALUES (?, ?, ?, ?)
        ''', (person_id, camera_id, location, datetime.now()))
        self.conn.commit()

    def add_alert(self, person_id, camera_id, reason):
        self.cursor.execute('''
            INSERT INTO alerts (person_id, camera_id, reason, timestamp) 
            VALUES (?, ?, ?, ?)
        ''', (person_id, camera_id, reason, datetime.now()))
        self.conn.commit()

    def get_recent_events(self, limit=50):
        self.cursor.execute('''
            SELECT person_id, camera_id, location, timestamp 
            FROM tracking_events 
            ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        return [{"person_id": row[0], "camera_id": row[1], "location": row[2], "timestamp": row[3]} for row in self.cursor.fetchall()]

    def list_active_persons(self):
        self.cursor.execute('''
            SELECT DISTINCT person_id FROM tracking_events
            WHERE timestamp >= datetime('now', '-10 minutes')
        ''')
        return [row[0] for row in self.cursor.fetchall()]

db = DatabaseManager()
