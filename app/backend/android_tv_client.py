"""Small async client for launching media through Android TV Remote v2."""

import asyncio
from pathlib import Path
from urllib.parse import urlparse
from androidtvremote2 import (
    AndroidTVRemote,
    CannotConnect,
    ConnectionClosed,
    InvalidAuth,
)


class AndroidTVError(RuntimeError):
    """Base error raised by the Android TV client."""


class AndroidTVConfigurationError(AndroidTVError):
    """The TV host or pairing certificate is not configured."""


class AndroidTVConnectionError(AndroidTVError):
    """The configured TV could not be reached or authenticated."""


class AndroidTVClient:
    """Launch links and send remote-control keys to an already-paired Android TV."""

    def __init__(
        self,
        host: str | None,
        cert_file: str,
        key_file: str,
        client_name: str = "Archie TV Remote",
        connect_timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.cert_file = Path(cert_file).expanduser()
        self.key_file = Path(key_file).expanduser()
        self.client_name = client_name
        self.connect_timeout = connect_timeout

    def _validate_configuration(self) -> None:
        if not self.host:
            raise AndroidTVConfigurationError("TV_HOST is not configured")
        if not self.cert_file.is_file() or not self.key_file.is_file():
            raise AndroidTVConfigurationError(
                "Android TV is not paired: TV_CERT_FILE or TV_KEY_FILE is missing"
            )

    async def launch_link(self, link: str) -> None:
        """Send a URI to Android TV and disconnect after it is queued."""
        self._validate_configuration()
        if not urlparse(link).scheme:
            raise AndroidTVConfigurationError("Media link must include a URI scheme")

        remote = AndroidTVRemote(
            self.client_name,
            str(self.cert_file),
            str(self.key_file),
            self.host,
        )
        try:
            await asyncio.wait_for(remote.async_connect(), timeout=self.connect_timeout)
            remote.send_launch_app_command(link)
            # send_launch_app_command buffers the protobuf message asynchronously.
            await asyncio.sleep(0.25)
        except TimeoutError as exc:
            raise AndroidTVConnectionError(
                "Timed out connecting to Android TV"
            ) from exc
        except InvalidAuth as exc:
            raise AndroidTVConnectionError(
                "Android TV pairing is no longer valid"
            ) from exc
        except (CannotConnect, ConnectionClosed, OSError) as exc:
            raise AndroidTVConnectionError(
                f"Could not connect to Android TV: {exc}"
            ) from exc
        finally:
            remote.disconnect()

    async def send_key(self, key_code: str) -> None:
        """Send one short remote-control key press to Android TV."""
        self._validate_configuration()

        remote = AndroidTVRemote(
            self.client_name,
            str(self.cert_file),
            str(self.key_file),
            self.host,
        )
        try:
            await asyncio.wait_for(remote.async_connect(), timeout=self.connect_timeout)
            remote.send_key_command(key_code)
            # send_key_command buffers the protobuf message asynchronously.
            await asyncio.sleep(0.25)
        except TimeoutError as exc:
            raise AndroidTVConnectionError(
                "Timed out connecting to Android TV"
            ) from exc
        except InvalidAuth as exc:
            raise AndroidTVConnectionError(
                "Android TV pairing is no longer valid"
            ) from exc
        except (CannotConnect, ConnectionClosed, OSError) as exc:
            raise AndroidTVConnectionError(
                f"Could not connect to Android TV: {exc}"
            ) from exc
        finally:
            remote.disconnect()
