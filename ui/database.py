import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('eye_care.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS screenings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_name TEXT,
                  date TEXT,
                  condition TEXT,
                  confidence REAL,
                  gradcam_url TEXT)''')
    conn.commit()
    conn.close()

def save_screening(name, condition, confidence, gradcam_url):
    conn = sqlite3.connect('eye_care.db')
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO screenings (patient_name, date, condition, confidence, gradcam_url) VALUES (?, ?, ?, ?, ?)",
              (name, date_str, condition, confidence, gradcam_url))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('eye_care.db')
    c = conn.cursor()
    c.execute("SELECT * FROM screenings ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows
