from enum import Enum
from typing import Union


class Url:
    """
    This class holds all information regarding where requests should be
    routed to within the scope of the Strava API

    Note that only those that I have made use of have been implemented
    """
    class APIVersion(str, Enum):
        v3 = "v3"

    def __init__(self, api_version: str = APIVersion.v3.value) -> None:
        if not api_version:
            raise ValueError("api_version must be specified")

        self.api_version = self.APIVersion(api_version).value
        self._base_api_path = "https://www.strava.com"

    @property
    def authentication_path(self) -> str:
        return f"{self._base_api_path}/api/{self.api_version}/oauth/token"

    @property
    def athlete_path(self) -> str:
        return f"{self._base_api_path}/api/{self.api_version}/athlete"

    @property
    def activities_path(self) -> str:
        return f"{self._base_api_path}/api/{self.api_version}/athlete/activities"

    def activity_stream_path(self, activity_id: Union[str, int]) -> str:
        return f"{self._base_api_path}/api/{self.api_version}/activities/{activity_id}/streams"
