"""Web and JSON entry points for intake and retrieval."""

import re
from collections.abc import Mapping
from pathlib import Path

from flask import (
    Blueprint, Flask, abort, current_app, jsonify, redirect, render_template, request, url_for,
)

from classifier import classify, clean_text, load_classifier, triage
from database import add_ticket, close_db, get_ticket, init_db, recent_tickets, resolve_ticket, ticket_summary

bp = Blueprint("tickets", __name__)
FIELDS = {
    "employee": ("Employee name", 100),
    "email": ("Work email", 254),
    "department": ("Department", 100),
    "title": ("Ticket title", 160),
    "description": ("Description", 5000),
}
EMAIL = re.compile(r"[^@\s]+@[^@\s.]+(?:\.[^@\s.]+)+")


def validate(payload):
    fields, errors = {}, {}
    for key, (label, limit) in FIELDS.items():
        value = payload.get(key)
        fields[key] = value.strip() if isinstance(value, str) else ""
        if not fields[key]:
            errors[key] = f"Enter {label.lower()}."
        elif len(fields[key]) > limit:
            errors[key] = f"Use {limit} characters or fewer."
    if "email" not in errors and not EMAIL.fullmatch(fields["email"]):
        errors["email"] = "Enter a valid work email address."
    impact = payload.get("business_impact") or None
    if impact not in (None, "Standard", "High"):
        errors["business_impact"] = "Choose Standard or High."
    fields["business_impact"] = impact
    if "title" not in errors and "description" not in errors:
        cleaned = clean_text(f'{fields["title"]} {fields["description"]}')
        if not any(character.isalpha() for character in cleaned):
            errors["description"] = "Describe the issue in words."
    return fields, errors


def submit(fields, *, description_only=False):
    text = fields["description"] if description_only else f'{fields["title"]}\n{fields["description"]}'
    category, confidence = classify(current_app.extensions["ticket_classifier"], text)
    impact = fields["business_impact"]
    severity, priority = triage(text, impact)
    return add_ticket(fields, category, severity, priority, confidence, impact or "Standard")


def serialize(ticket):
    title = " ".join(ticket["title"].split())
    return dict(ticket) | {"ticket_id": f'T{ticket["id"]:04d}', "display_title": title}


def lookup(ticket_id):
    # SQLite integer keys are signed 64-bit.
    return get_ticket(ticket_id) if ticket_id <= 2**63 - 1 else None


@bp.route("/", methods=["GET", "POST"])
def form():
    fields, errors = {}, {}
    if request.method == "POST":
        fields, errors = validate(request.form)
        if not errors:
            ticket = submit(fields)
            return redirect(url_for(".detail", ticket_id=ticket["id"]), code=303)
    return render_template("index.html", fields=fields, errors=errors, specs=FIELDS), (
        422 if errors else 200
    )


@bp.get("/tickets")
def listing():
    return render_template("list.html", tickets=map(serialize, recent_tickets()))


@bp.get("/overview")
def overview():
    return render_template("overview.html", summary=ticket_summary(),
                           tickets=[serialize(ticket) for ticket in recent_tickets()][:5])


@bp.get("/tickets/<int:ticket_id>")
def detail(ticket_id):
    ticket = lookup(ticket_id)
    if ticket is None:
        abort(404)
    return render_template("detail.html", ticket=serialize(ticket))


@bp.post("/tickets/<int:ticket_id>/resolve")
def resolve(ticket_id):
    if lookup(ticket_id) is None:
        abort(404)
    resolve_ticket(ticket_id)
    return redirect(url_for(".detail", ticket_id=ticket_id), code=303)


@bp.post("/ticket")
def api_create():
    if not request.is_json:
        return jsonify(error="Send application/json."), 415
    payload = request.get_json(silent=True)
    if not isinstance(payload, Mapping):
        return jsonify(error="Send a valid JSON object."), 400
    if set(payload) == {"description"}:
        description = payload["description"]
        if not isinstance(description, str) or not description.strip() or len(description) > 5000:
            return jsonify(error="Enter a description of 1–5000 characters."), 422
        fields = dict.fromkeys(FIELDS, "")
        fields.update(description=description.strip(), title="IT support ticket",
                      business_impact=None)
        ticket = submit(fields, description_only=True)
        return jsonify(ticket=ticket["description"], category=ticket["category"],
                       severity=ticket["severity"], priority=ticket["priority"],
                       status=ticket["status"])

    fields, errors = validate(payload)
    if errors:
        return jsonify(error="Check the ticket fields.", fields=errors), 422
    ticket = serialize(submit(fields))
    return jsonify(ticket), 201, {"Location": url_for(".api_detail", ticket_id=ticket["id"])}


@bp.get("/api/tickets")
def api_list():
    return jsonify(tickets=[serialize(ticket) for ticket in recent_tickets()])


@bp.get("/api/tickets/<int:ticket_id>")
def api_detail(ticket_id):
    ticket = lookup(ticket_id)
    if ticket is None:
        return jsonify(error="Ticket not found."), 404
    return jsonify(serialize(ticket))


def create_app(config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=Path(__file__).resolve().with_name("tickets.db"),
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    if config is not None:
        app.config.update(config)
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()
    app.extensions["ticket_classifier"] = load_classifier()
    app.register_blueprint(bp)

    @app.errorhandler(413)
    def request_too_large(_error):
        message = "The request exceeds the 64 KiB limit."
        if request.path == "/ticket":
            return jsonify(error=message), 413
        return render_template("error.html", message=message), 413

    return app


if __name__ == "__main__":
    create_app().run()
