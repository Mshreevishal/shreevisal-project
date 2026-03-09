import sqlite3

def create_tables():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY,
        name TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance(
        student_id INTEGER,
        name TEXT,
        date TEXT,
        time TEXT
    )
    """)

    conn.commit()
    conn.close()


def add_student(student_id,name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("INSERT INTO students VALUES (?,?)",(student_id,name))

    conn.commit()
    conn.close()