"""Manage the tickets.db SQLite database."""

import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY,
    employee TEXT NOT NULL,
    email TEXT NOT NULL,
    department TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    priority TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    business_impact TEXT NOT NULL DEFAULT 'Standard',
    status TEXT NOT NULL DEFAULT 'Open',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None):
    database = g.pop("db", None)
    if database is not None:
        database.close()


def init_db():
    database = get_db()
    with database:
        database.execute("BEGIN IMMEDIATE")
        database.execute(SCHEMA)
        columns = {row["name"] for row in database.execute("PRAGMA table_info(tickets)")}
        if "business_impact" not in columns:
            database.execute("ALTER TABLE tickets ADD business_impact TEXT NOT NULL DEFAULT 'Standard'")
        if "status" not in columns:
            database.execute("ALTER TABLE tickets ADD status TEXT NOT NULL DEFAULT 'Open'")


def add_ticket(fields, category, severity, priority, confidence, business_impact):
    database = get_db()
    with database:
        cursor = database.execute(
            """INSERT INTO tickets
               (employee, email, department, title, description, category, severity,
                priority, confidence, business_impact)
               VALUES (:employee, :email, :department, :title, :description, :category,
                       :severity, :priority, :confidence, :business_impact)""",
            dict(fields, category=category, severity=severity, priority=priority,
                 confidence=confidence, business_impact=business_impact),
        )
        ticket = get_ticket(cursor.lastrowid)
    return ticket


def get_ticket(ticket_id):
    return get_db().execute(
        "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
    ).fetchone()


def recent_tickets():
    return get_db().execute("SELECT * FROM tickets ORDER BY id DESC LIMIT 50").fetchall()


def resolve_ticket(ticket_id):
    database = get_db()
    with database:
        database.execute("UPDATE tickets SET status = 'Closed' WHERE id = ?", (ticket_id,))


def ticket_summary():
    return get_db().execute(
        """SELECT COUNT(*) AS total,
           COALESCE(SUM(status = 'Open'), 0) AS open_count,
           COALESCE(SUM(status = 'Closed'), 0) AS closed_count,
           COALESCE(SUM(severity = 'Critical' AND status = 'Open'), 0) AS critical_count
           FROM tickets"""
    ).fetchone()
