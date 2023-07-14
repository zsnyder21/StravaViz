import datetime
import json
import pickle
import requests

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Union

from src.urls import Url


class StravaClient:
    """
    This class is responsible for interfacing with Strava's API
    """
    def __init__(self, client_id: str, client_secret: str, authorization_code: str = None, refresh_token: str = None):
        if refresh_token is None and authorization_code is None:
            raise ValueError("At least a refresh token or authorization code must be specified")
        self._client_id = client_id
        self._client_secret = client_secret
        self._authorization_code = authorization_code
        self._urls = Url(api_version=Url.APIVersion("v3"))
        self._access_token = None
        self._access_token_expiry = None
        self._refresh_token = refresh_token

    def extract_athlete(self) -> dict:
        """
        Extracts a dictionary with the athlete information associated with the authentication credentials being used

        :return: A dictionary containing athlete information
        """
        return self._make_request(
            method="GET",
            url=self._urls.athlete_path
        )

    def extract_activities(self, file: str = None):
        """
        Extracts a list of all activities

        :param file: Location to export the activities to
        :return: A list of dictionaries containing all activities
        """
        activities = []
        activities_per_page = 200  # 200 is the internal limit of activities per page
        page = 1
        response_count = -1

        while response_count != 0:
            activities_temp = self._make_request(
                method="GET",
                url=self._urls.activities_path,
                params={"per_page": activities_per_page, "page": page}
            )

            response_count = len(activities_temp)
            activities.extend(activities_temp)
            page += 1

        if file is not None:
            with open(file, "wb") as f:
                pickle.dump(obj=activities, file=f)

        return activities

    def extract_activity_streams(self, activity_ids: int, file: str = None):
        """
        Extracts activity streams for the passed activity id

        :param activity_ids: Activity ids of the streams we wish to fetch
        :param file: Location to save the streams
        :return:
        """
        if not isinstance(activity_ids, Iterable):
            activity_ids = [activity_ids]
        activity_streams = []

        for activity_id in activity_ids:
            stream = self._make_request(
                method="GET",
                url=self._urls.activity_stream_path(activity_id=activity_id),
                params={"keys": "time,latlng,altitude"}  # Docs say array, but this doesn't return streams as expected
            )

            for substream in stream:
                substream["activity_id"] = activity_id

            activity_streams.append(stream)

        if file is not None:
            with open(file, "wb") as f:
                pickle.dump(activity_streams, f)

        return activity_streams

    def _authenticate(self):
        """
        Authenticates with Strava's API using either a refresh token or an authorization code
        depending on what is available
        """
        headers = {"Content-Type": "application/json"}
        if self._refresh_token:
            data = {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token"
            }
        else:
            data = {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "code": self._authorization_code,
                "grant_type": "authorization_code"
            }

        login_response = requests.post(
            self._urls.authentication_path,
            headers=headers,
            data=json.dumps(data)
        )
        login_data = json.loads(login_response.text)
        if login_response.status_code != 200:
            raise ConnectionError(
                f"Authentication failed - {login_data.get('message')} - {login_data.get('errors')}"
            )
        self._access_token = login_data["access_token"]
        self._access_token_expiry = login_data["expires_at"]
        self._refresh_token = login_data["refresh_token"]

    def _is_token_expired(self) -> bool:
        """
        Return true if the token is nearly expired (expires in 60s or less)

        :return: Boolean indicating whether our access token is about to expire
        """
        if self._access_token is None:
            return True

        current_timestamp = int(datetime.timestamp(datetime.now(tz=timezone.utc)))
        return self._access_token_expiry - current_timestamp < 60

    def _should_refresh(self) -> bool:
        """
        Return true if we should refresh the access token. If False, authentication is required

        :return: Boolean indicating whether we should refresh the access token
        """
        return self._refresh_token is not None and self._is_token_expired()

    def _is_authenticated(self) -> bool:
        """
        Returns true if we are authenticated

        :return: Boolean flag indicating whether we are authenticated
        """
        if self._access_token is None:
            return False
        if self._refresh_token is None:
            return False

        return not self._is_token_expired()

    def requires_auth(f):
        """
        Decorator that guarantees the calling function has a valid access token before making requests
        """
        def auth_wrapper(self, *args, **kwargs):
            if self._should_refresh() or not self._is_authenticated():
                self._authenticate()

            return f(self, *args, **kwargs)

        return auth_wrapper

    @requires_auth
    def _make_request(self, method: str, url: str, data: Union[dict, list, str] = None, params: dict = None):
        """
        This is the method that handles sending out HTTP requests to Strava

        :param method: Requests method to use (e.g. GET, PUT, POST, etc)
        :param url: The URL to make the request to
        :param data: Data to send with the request
        :param params: Parameters to send with the request
        :return: Returns a JSON representation of the response returned by the request
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_token}"
        }

        if method.upper() == "GET":
            response = requests.get(
                url=url,
                headers=headers,
                params=params
            )
        else:
            raise NotImplementedError(
                "This request method is not yet supported. Implement before using"
            )

        if response.status_code != 200:
            raise ConnectionError(
                f"Failure while making request: Status code {response.status_code} - {response.text}"
            )

        return json.loads(response.text)
