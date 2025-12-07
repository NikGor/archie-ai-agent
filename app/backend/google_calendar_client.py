"""Google Calendar API client for event management."""

import datetime
import logging
import os
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
DEFAULT_TIMEZONE = "Europe/Berlin"


class GoogleCalendarClient:
    """
    Wrapper for Google Calendar API operations.
    Uses Service Account for server-to-server communication.
    """

    def __init__(
        self,
        credentials_file: str | None = None,
        timezone: str = DEFAULT_TIMEZONE,
    ):
        self.credentials_file = credentials_file or os.getenv(
            "GOOGLE_CALENDAR_CREDENTIALS_FILE",
            "credentials.json",
        )
        self.timezone = timezone
        self.service = self._authenticate()

    def _authenticate(self):
        """Authenticates using the Service Account file."""
        if not os.path.exists(self.credentials_file):
            logger.error(
                f"gcal_auth_001: Credentials file not found: "
                f"\033[31m{self.credentials_file}\033[0m"
            )
            raise FileNotFoundError(
                f"Credentials file {self.credentials_file} not found"
            )
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_file,
            scopes=SCOPES,
        )
        logger.info(
            f"gcal_auth_002: Authenticated with credentials: "
            f"\033[36m{self.credentials_file}\033[0m"
        )
        return build("calendar", "v3", credentials=credentials)

    async def list_events(
        self,
        calendar_id: str = "primary",
        time_min: str | None = None,
        time_max: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """
        Lists events for the specified calendar.

        Args:
            calendar_id: Calendar ID or 'primary' for default calendar
            time_min: Start time filter in ISO format
            time_max: End time filter in ISO format
            max_results: Maximum number of events to return

        Returns:
            Dict with success status and events list
        """
        try:
            if not time_min:
                time_min = datetime.datetime.utcnow().isoformat() + "Z"
            request_params = {
                "calendarId": calendar_id,
                "timeMin": time_min,
                "maxResults": max_results,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if time_max:
                request_params["timeMax"] = time_max
            events_result = self.service.events().list(**request_params).execute()
            events = events_result.get("items", [])
            logger.info(f"gcal_list_001: Retrieved \033[33m{len(events)}\033[0m events")
            return {
                "success": True,
                "events": [self._format_event(event) for event in events],
                "count": len(events),
            }
        except HttpError as error:
            logger.error(f"gcal_list_error_001: \033[31m{error}\033[0m")
            return {"success": False, "error": str(error)}

    async def get_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """
        Gets a single event by ID.

        Args:
            event_id: The event ID
            calendar_id: Calendar ID or 'primary' for default calendar

        Returns:
            Dict with success status and event data
        """
        try:
            event = (
                self.service.events()
                .get(
                    calendarId=calendar_id,
                    eventId=event_id,
                )
                .execute()
            )
            logger.info(f"gcal_get_001: Retrieved event \033[36m{event_id}\033[0m")
            return {
                "success": True,
                "event": self._format_event(event),
            }
        except HttpError as error:
            logger.error(f"gcal_get_error_001: \033[31m{error}\033[0m")
            return {"success": False, "error": str(error)}

    async def create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        calendar_id: str = "primary",
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Creates a new event.

        Args:
            summary: Event title
            start_time: Start time in ISO format
            end_time: End time in ISO format
            calendar_id: Calendar ID or 'primary' for default calendar
            description: Event description
            location: Event location
            attendees: List of attendee email addresses

        Returns:
            Dict with success status and created event data
        """
        event_body: dict[str, Any] = {
            "summary": summary,
            "start": {
                "dateTime": start_time,
                "timeZone": self.timezone,
            },
            "end": {
                "dateTime": end_time,
                "timeZone": self.timezone,
            },
        }
        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location
        if attendees:
            event_body["attendees"] = [{"email": email} for email in attendees]
        try:
            event = (
                self.service.events()
                .insert(
                    calendarId=calendar_id,
                    body=event_body,
                )
                .execute()
            )
            logger.info(
                f"gcal_create_001: Created event \033[36m{event.get('id')}\033[0m"
            )
            return {
                "success": True,
                "event": self._format_event(event),
                "html_link": event.get("htmlLink"),
            }
        except HttpError as error:
            logger.error(f"gcal_create_error_001: \033[31m{error}\033[0m")
            return {"success": False, "error": str(error)}

    async def update_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        summary: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Updates an existing event using PATCH (partial update).

        Args:
            event_id: The event ID to update
            calendar_id: Calendar ID or 'primary' for default calendar
            summary: New event title
            start_time: New start time in ISO format
            end_time: New end time in ISO format
            description: New event description
            location: New event location
            attendees: New list of attendee email addresses

        Returns:
            Dict with success status and updated event data
        """
        patch_body: dict[str, Any] = {}
        if summary is not None:
            patch_body["summary"] = summary
        if description is not None:
            patch_body["description"] = description
        if location is not None:
            patch_body["location"] = location
        if start_time is not None:
            patch_body["start"] = {
                "dateTime": start_time,
                "timeZone": self.timezone,
            }
        if end_time is not None:
            patch_body["end"] = {
                "dateTime": end_time,
                "timeZone": self.timezone,
            }
        if attendees is not None:
            patch_body["attendees"] = [{"email": email} for email in attendees]
        if not patch_body:
            return {"success": False, "error": "No fields to update"}
        try:
            event = (
                self.service.events()
                .patch(
                    calendarId=calendar_id,
                    eventId=event_id,
                    body=patch_body,
                )
                .execute()
            )
            logger.info(f"gcal_update_001: Updated event \033[36m{event_id}\033[0m")
            return {
                "success": True,
                "event": self._format_event(event),
            }
        except HttpError as error:
            logger.error(f"gcal_update_error_001: \033[31m{error}\033[0m")
            return {"success": False, "error": str(error)}

    async def delete_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """
        Deletes an event by ID.

        Args:
            event_id: The event ID to delete
            calendar_id: Calendar ID or 'primary' for default calendar

        Returns:
            Dict with success status
        """
        try:
            self.service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
            ).execute()
            logger.info(f"gcal_delete_001: Deleted event \033[36m{event_id}\033[0m")
            return {"success": True, "message": f"Event {event_id} deleted"}
        except HttpError as error:
            logger.error(f"gcal_delete_error_001: \033[31m{error}\033[0m")
            return {"success": False, "error": str(error)}

    async def get_today_events(
        self,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """
        Gets all events for today.

        Args:
            calendar_id: Calendar ID or 'primary' for default calendar

        Returns:
            Dict with success status and today's events
        """
        today = datetime.date.today()
        time_min = (
            datetime.datetime.combine(
                today,
                datetime.time.min,
            ).isoformat()
            + "Z"
        )
        time_max = (
            datetime.datetime.combine(
                today,
                datetime.time.max,
            ).isoformat()
            + "Z"
        )
        return await self.list_events(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=time_max,
            max_results=50,
        )

    async def get_events_for_date(
        self,
        date: str,
        calendar_id: str = "primary",
    ) -> dict[str, Any]:
        """
        Gets all events for a specific date.

        Args:
            date: Date in YYYY-MM-DD format
            calendar_id: Calendar ID or 'primary' for default calendar

        Returns:
            Dict with success status and events for the date
        """
        target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        time_min = (
            datetime.datetime.combine(
                target_date,
                datetime.time.min,
            ).isoformat()
            + "Z"
        )
        time_max = (
            datetime.datetime.combine(
                target_date,
                datetime.time.max,
            ).isoformat()
            + "Z"
        )
        return await self.list_events(
            calendar_id=calendar_id,
            time_min=time_min,
            time_max=time_max,
            max_results=50,
        )

    def _format_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Formats a raw Google Calendar event into a cleaner structure."""
        start = event.get("start", {})
        end = event.get("end", {})
        return {
            "id": event.get("id"),
            "title": event.get("summary"),
            "description": event.get("description"),
            "location": event.get("location"),
            "start_time": start.get("dateTime") or start.get("date"),
            "end_time": end.get("dateTime") or end.get("date"),
            "html_link": event.get("htmlLink"),
            "attendees": [
                attendee.get("email") for attendee in event.get("attendees", [])
            ],
            "status": event.get("status"),
            "created": event.get("created"),
            "updated": event.get("updated"),
        }
