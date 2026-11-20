import sqlite3

import pytest

from app import create_app
from classifier import CATEGORIES, classify, clean_text, load_dataset, triage
from database import SCHEMA
from evaluation import evaluate


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    database = tmp_path_factory.mktemp("supportpilot") / "test.sqlite3"
    return create_app({"TESTING": True, "DATABASE": database})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def ticket():
    return {
        "employee": "Arun",
        "email": "arun@example.com",
        "department": "Sales",
        "title": "VPN not working",
        "description": (
            "I am unable to connect to the company VPN. "
            "I have an important client meeting in 30 minutes."
        ),
    }


def test_preprocessing():
    assert clean_text("URGENT!!! My VPN is NOT connecting!!! Please help me ASAP!!!") == (
        "urgent vpn not connecting please help"
    )
    assert clean_text("VPN isn't working. user@example.org https://example.org/path") == (
        "vpn not working"
    )


def test_kaggle_training_coverage():
    texts, labels = load_dataset()
    assert len(texts) == 38269
    assert set(labels) == CATEGORIES


def test_classification_uses_kaggle_labels(app):
    label, confidence = classify(app.extensions["ticket_classifier"], "VPN connection access request")
    assert label in CATEGORIES
    assert 0 < confidence < 1


@pytest.mark.parametrize(("text", "expected"), [
    ("Production down and I have a client meeting", ("Critical", "P1")),
    ("The entire company has no internet", ("Critical", "P1")),
    ("My team cannot work", ("High", "P2")),
    ("This is urgent", ("High", "P2")),
    ("My VPN shows an error", ("Medium", "P3")),
    ("I forgot my password.", ("Low", "P4")),
    ("How do I install the approved editor?", ("Low", "P4")),
    ("The browser shows an error.", ("Medium", "P3")),
])
def test_triage(text, expected):
    assert triage(text) == expected


def test_api_create_and_retrieve(client, ticket, app):
    response = client.post("/ticket", json=ticket)
    assert response.status_code == 201
    result = response.get_json()
    assert result["ticket_id"] == f'T{result["id"]:04d}'
    assert result["category"] in CATEGORIES
    assert result["severity"] == "High"
    assert result["priority"] == "P2"
    assert result["status"] == "Open"
    assert result["business_impact"] == "Standard"
    assert 0 < result["confidence"] < 1
    assert result["created_at"].endswith("Z")
    with sqlite3.connect(app.config["DATABASE"]) as database:
        row = database.execute(
            "SELECT email, priority FROM tickets WHERE id = ?", (result["id"],)
        ).fetchone()
    assert row == (ticket["email"], "P2")
    assert client.get(response.headers["Location"]).get_json() == result
    assert result in client.get("/api/tickets").get_json()["tickets"]


def test_form_create_and_view(client, ticket):
    form = client.get("/")
    assert form.status_code == 200
    assert b"bootstrap.min.css" in form.data
    assert client.get("/static/bootstrap.min.css").status_code == 200
    response = client.post("/", data=ticket)
    assert response.status_code == 303
    detail = client.get(response.headers["Location"])
    assert detail.status_code == 200
    assert b"VPN" in detail.data
    assert b"Category confidence" in detail.data
    assert b"VPN not working" in client.get("/tickets").data


def test_invalid_fields_do_not_create_ticket(client, ticket):
    before = client.get("/api/tickets").get_json()
    api = client.post("/ticket", json=dict(ticket, email="bad", title=None))
    assert api.status_code == 422
    assert set(api.get_json()["fields"]) == {"email", "title"}
    form = client.post("/", data=dict(ticket, email="bad"))
    assert form.status_code == 422
    assert b"Enter a valid work email address." in form.data
    assert client.get("/api/tickets").get_json() == before


@pytest.mark.parametrize(("body", "content_type", "status"), [
    ("oops", "text/plain", 415),
    ("{", "application/json", 400),
    ("[]", "application/json", 400),
    ("x" * 65537, "application/json", 413),
])
def test_bad_api_requests(client, body, content_type, status):
    response = client.post("/ticket", data=body, content_type=content_type)
    assert response.status_code == status
    assert "error" in response.get_json()


def test_missing_ticket(client):
    assert client.get("/tickets/9999999999").status_code == 404
    assert client.get("/api/tickets/999999999999999999999").get_json() == {
        "error": "Ticket not found."
    }


def test_impact_changes_priority(client, ticket):
    result = client.post("/ticket", json=dict(ticket, business_impact="High")).get_json()
    assert (result["severity"], result["priority"], result["business_impact"]) == (
        "High", "P1", "High"
    )
    invalid = client.post("/ticket", json=dict(ticket, business_impact="Unknown"))
    assert invalid.status_code == 422
    assert "business_impact" in invalid.get_json()["fields"]


def test_pdf_description_only_request(client):
    description = "VPN is not connecting and I cannot attend my client meeting"
    response = client.post("/ticket", json={"description": description})
    assert response.status_code == 200
    result = response.get_json()
    assert result["category"] in CATEGORIES
    assert result == {
        "ticket": description,
        "category": result["category"],
        "severity": "High",
        "priority": "P2",
        "status": "Open",
    }


def test_priority_examples_use_the_pdf_formulas():
    assert triage("server down") == ("Critical", "P1")
    assert triage("server down", "High") == ("Critical", "P1")
    assert triage("server down", "Standard") == ("Critical", "P4")
    assert triage("client meeting", "High") == ("High", "P1")


def test_resolve_ticket_persists_and_is_repeatable(client, ticket):
    created = client.post("/ticket", json=ticket).get_json()
    path = f"/tickets/{created['id']}/resolve"
    assert client.get(path).status_code == 405
    assert client.post(path).status_code == 303
    assert client.get(f"/api/tickets/{created['id']}").get_json()["status"] == "Closed"
    assert client.post(path).status_code == 303
    detail = client.get(f"/tickets/{created['id']}").data
    assert b"This ticket is closed." in detail
    assert b"Mark as resolved" not in detail
    assert client.post("/tickets/999999999/resolve").status_code == 404


def test_legacy_database_migration(tmp_path, app):
    path = tmp_path / "legacy.sqlite3"
    schema = "\n".join(line for line in SCHEMA.splitlines()
                       if "business_impact" not in line and "status" not in line)
    with sqlite3.connect(path) as database:
        database.execute(schema)
        database.execute(
            """INSERT INTO tickets (employee, email, department, title, description,
               category, severity, priority, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("Arun", "arun@example.com", "Sales", "VPN", "Unable to connect",
             "VPN", "High", "P2", 0.4),
        )
    migrated = create_app({"TESTING": True, "DATABASE": path})
    result = migrated.test_client().get("/api/tickets/1").get_json()
    assert (result["title"], result["priority"], result["status"], result["business_impact"]) == (
        "VPN", "P2", "Open", "Standard"
    )


def test_evaluation_uses_kaggle_holdout(app, capsys):
    evaluate(app.extensions["ticket_classifier"])
    report = capsys.readouterr().out
    assert "9568 separate labeled tickets" in report
    assert "Category:" in report and "Severity:" in report
