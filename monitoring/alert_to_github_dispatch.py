"""Alertmanager bridge that dispatches GitHub repository events.

This tiny HTTP server receives Alertmanager webhooks and dispatches
`repository_dispatch` events to trigger retraining workflows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib import request


BANNER = "=" * 50


def print_banner(title: str) -> None:
    """Print a consistent section banner."""
    print(BANNER)
    print(title)
    print(BANNER)


def parse_event_type(payload: dict) -> str:
    """Map alert labels to repository dispatch event types."""
    alerts = payload.get("alerts", [])
    for alert in alerts:
        name = str(alert.get("labels", {}).get("alertname", ""))
        if name == "FraudRecallLow":
            return "fraud_recall_drop"
        if name == "DataDriftHigh":
            return "fraud_drift_exceeded"
    return "fraud_recall_drop"


def dispatch_to_github(repository: str, token: str, event_type: str, payload: dict) -> None:
    """Send repository_dispatch event to GitHub."""
    url = f"https://api.github.com/repos/{repository}/dispatches"
    body = json.dumps(
        {
            "event_type": event_type,
            "client_payload": {
                "source": "alertmanager",
                "alerts": payload.get("alerts", []),
            },
        }
    ).encode("utf-8")
    req = request.Request(url=url, method="POST", data=body)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")
    with request.urlopen(req, timeout=10) as response:
        if response.status not in (200, 204):
            raise RuntimeError(f"GitHub dispatch failed with status {response.status}")


class AlertHandler(BaseHTTPRequestHandler):
    """Handle alert payloads and trigger repository dispatches."""

    repository = ""
    token = ""

    def do_POST(self) -> None:  # noqa: N802
        """Process Alertmanager webhook payloads."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8") or "{}")
            if not self.token:
                raise RuntimeError("GITHUB_TOKEN is not set")
            event_type = parse_event_type(payload)
            dispatch_to_github(self.repository, self.token, event_type, payload)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        except Exception as exc:  # pragma: no cover - runtime safety
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        """Keep logs concise in container output."""
        print("alert-bridge:", format % args)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Alertmanager GitHub bridge server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5001, help="Bind port")
    return parser.parse_args()


def main() -> int:
    """Start the webhook bridge service."""
    args = parse_args()
    repository = os.getenv("GITHUB_REPOSITORY", "")
    token = os.getenv("GITHUB_TOKEN", "")
    if not repository:
        print("[ERROR] GITHUB_REPOSITORY is required", file=sys.stderr)
        return 1

    AlertHandler.repository = repository
    AlertHandler.token = token
    print_banner("Alertmanager to GitHub Dispatch Bridge")
    print(f"Repository: {repository}")
    print(f"Listening on {args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), AlertHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
