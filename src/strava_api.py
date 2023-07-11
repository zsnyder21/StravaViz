import datetime
import json
import requests

from datetime import datetime, timezone
from typing import Union

from src.urls import Url


class StravaClient:
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

    def extract_athlete(self):
        return self._make_request(
            method="GET",
            url=self._urls.athlete_path
        )

    def extract_activities(self):
        activities = []
        page = 1
        response_count = -1

        while response_count != 0:
            # 200 is the internal limit of activities per page
            activities_temp = self._make_request(
                method="GET",
                url=self._urls.activities_path,
                params={"per_page": 200, "page": page}
            )

            response_count = len(activities_temp)
            activities.extend(activities_temp)
            page += 1

        return activities

    def _authenticate(self):
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
        :return: Boolean indicating whether or not our access token is about to expire
        """
        if self._access_token is None:
            return True

        current_timestamp = int(datetime.timestamp(datetime.now(tz=timezone.utc)))
        return self._access_token_expiry - current_timestamp < 60

    def _should_refresh(self) -> bool:
        """
        Return true if we should refresh the access token. If False, authentication is required
        :return: Boolean indicating whether or not we should refresh the access token
        """
        return self._refresh_token is not None and self._is_token_expired()


    def _is_authenticated(self) -> bool:
        """
        Returns true if we are authenticated
        :return: Boolean flag indicating whether or not we are authenticated
        """
        if self._access_token is None:
            return False
        if self._refresh_token is None:
            return False

        return not self._is_token_expired()

    def requires_auth(f):
        """
        Decorator that guarantees the calling function has a valid access token before making requests
        :return:
        """
        def auth_wrapper(self, *args, **kwargs):
            if self._should_refresh() or not self._is_authenticated():
                self._authenticate()

            return f(self, *args, **kwargs)

        return auth_wrapper

    @requires_auth
    def _make_request(self, method: str, url: str, data: Union[dict, list, str] = None, params: dict = None):
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


def main():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    authorization_code = os.getenv("AUTHORIZATION_CODE")
    refresh_token = os.getenv("REFRESH_TOKEN")

    client = StravaClient(
        client_id=client_id,
        client_secret=client_secret,
        authorization_code=authorization_code,
        refresh_token=refresh_token
    )

    activities = client.extract_activities()
    print(activities[-1])


if __name__ == "__main__":
    main()

