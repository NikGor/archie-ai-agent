"""Gmail API client for read-only mailbox access."""

import asyncio
import base64
import logging
import os
from datetime import datetime
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from app.config import settings


logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
METADATA_HEADERS = ["From", "Subject", "Date"]
MAX_BODY_CHARS = 4000


class GmailClient:
    """
    Wrapper for Gmail API read operations.
    Uses OAuth user credentials (token_gmail.json with a refresh token).
    """

    def __init__(self, token_file: str | None = None):
        self.token_file = token_file or settings.gmail_token_file
        self.service = self._authenticate()

    def _authenticate(self):
        """Authenticates using the saved OAuth token file."""
        if not os.path.exists(self.token_file):
            logger.error(
                f"gmail_auth_001: Token file not found: \033[31m{self.token_file}\033[0m"
            )
            raise FileNotFoundError(f"Gmail token file {self.token_file} not found")
        credentials = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        logger.info(f"gmail_auth_002: Authenticated with token: \033[36m{self.token_file}\033[0m")
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    async def search_messages(self, query: str = "", max_results: int = 5) -> dict[str, Any]:
        """Searches messages with Gmail query syntax; empty query lists the inbox."""
        try:
            return await asyncio.to_thread(self._search_messages, query, max_results)
        except HttpError as error:
            logger.error(f"gmail_search_error_001: \033[31m{error}\033[0m")
            return {"success": False, "error": str(error)}

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Gets a single message with its plain-text body."""
        try:
            return await asyncio.to_thread(self._get_message, message_id)
        except HttpError as error:
            logger.error(f"gmail_get_error_001: \033[31m{error}\033[0m")
            return {"success": False, "error": str(error)}

    def _search_messages(self, query: str, max_results: int) -> dict[str, Any]:
        request_params: dict[str, Any] = {"userId": "me", "maxResults": max_results}
        if query:
            request_params["q"] = query
        else:
            request_params["labelIds"] = ["INBOX"]
        refs = self.service.users().messages().list(**request_params).execute().get("messages", [])
        messages = [
            self._format_message(
                self.service.users()
                .messages()
                .get(userId="me", id=ref["id"], format="metadata", metadataHeaders=METADATA_HEADERS)
                .execute()
            )
            for ref in refs
        ]
        logger.info(f"gmail_search_001: Retrieved \033[33m{len(messages)}\033[0m messages")
        return {"success": True, "messages": messages, "count": len(messages)}

    def _get_message(self, message_id: str) -> dict[str, Any]:
        raw = self.service.users().messages().get(userId="me", id=message_id, format="full").execute()
        message = self._format_message(raw)
        message["body"] = self._extract_body(raw.get("payload", {}))[:MAX_BODY_CHARS]
        return {"success": True, "message": message}

    @staticmethod
    def _format_message(raw: dict[str, Any]) -> dict[str, Any]:
        """Converts a raw Gmail message into a flat dict."""
        headers = {
            header["name"].lower(): header["value"]
            for header in raw.get("payload", {}).get("headers", [])
        }
        sender_name, sender_email = parseaddr(headers.get("from", ""))
        received_at = ""
        if headers.get("date"):
            received_at = GmailClient._format_date(headers["date"])
        parts = raw.get("payload", {}).get("parts", [])
        return {
            "message_id": raw["id"],
            "thread_id": raw.get("threadId", ""),
            "subject": headers.get("subject") or "(no subject)",
            "sender_name": sender_name or sender_email,
            "sender_email": sender_email,
            "snippet": raw.get("snippet", ""),
            "received_at": received_at,
            "is_unread": "UNREAD" in raw.get("labelIds", []),
            "has_attachments": any(part.get("filename") for part in parts),
            "gmail_url": f"https://mail.google.com/mail/u/0/#all/{raw['id']}",
        }

    @staticmethod
    def _format_date(date_header: str) -> str:
        try:
            parsed: datetime = parsedate_to_datetime(date_header)
        except (TypeError, ValueError):
            return date_header
        return parsed.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _extract_body(payload: dict[str, Any]) -> str:
        """Returns the first text/plain body found in the payload tree."""
        if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        for part in payload.get("parts", []):
            body = GmailClient._extract_body(part)
            if body:
                return body
        return ""
