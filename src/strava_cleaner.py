import gpxpy
import pandas as pd

from datetime import timedelta
from tqdm.auto import tqdm


class StravaCleaner:
    """
    This class takes raw responses from the Strava API and returns curated data
    """
    @staticmethod
    def clean_activities(activities: list, progress_bar: bool = True, file: str = None) -> pd.DataFrame:
        """
        Cleans activity data

        :param activities: List of raw activities returned by the Strava API
        :param progress_bar: Whether to display a progress bar
        :param file: Location to export the cleaned data to
        :return: Pandas DataFrame containing the curated strava data
        """
        activities = activities.copy()
        activities_clean = []
        with tqdm(total=len(activities), disable=not progress_bar) as pbar:
            for activity in activities:
                pbar.set_description(f"Processing {activity['name']}")
                activity_clean = {
                    "activity_id": activity["id"],
                    "name": activity["name"],
                    "type": activity["type"],
                    "distance": activity["distance"],
                    "moving_time": activity["moving_time"],
                    "elapsed_time": activity["elapsed_time"],
                    "elevation_gain": activity["total_elevation_gain"],
                    "start_date": pd.to_datetime(activity["start_date"]),
                    "start_date_local": pd.to_datetime(activity["start_date_local"]),
                    "flagged": activity["flagged"],
                    "start_latitude": (activity.get("start_latlng") or [None, None])[0],
                    "start_longitude": (activity["start_latlng"] or [None, None])[1],
                    "end_latitude": (activity["end_latlng"] or [None, None])[0],
                    "end_longitude": (activity["end_latlng"] or [None, None])[1],
                    "average_speed": activity["average_speed"],
                    "max_speed": activity["max_speed"],
                    "average_heartrate": activity.get("average_heartrate"),
                    "max_heartrate": activity.get("max_heartrate"),
                    "min_elevation": activity.get("elev_low"),
                    "max_elevation": activity.get("elev_high"),
                    "kilojoules": activity.get("kilojoules"),
                    "suffer_score": activity.get("suffer_score"),
                    "map_polyline": activity["map"]["summary_polyline"]
                }

                activities_clean.append(activity_clean)
                pbar.update(1)

            activities_clean = pd.DataFrame(activities_clean)
            if file:
                activities_clean.to_pickle(file)

            return activities_clean

    @staticmethod
    def clean_activity_streams(activity_streams: list, progress_bar: bool = True, file: str = None) -> pd.DataFrame:
        """
        Cleans activity streams into a Pandas DataFrame

        :param activity_streams: List of activity streams to process
        :param progress_bar: Boolean indicating whether a progress bar is displayed
        :param file: Location to save the cleaned data to
        :return: Pandas DataFrame containing activity stream information
        """
        streams = []

        with tqdm(total=len(activity_streams)) as pbar:
            for idx, activity_stream in enumerate(activity_streams):
                pbar.set_description(f"Processing stream {activity_stream[0]['activity_id']}")
                data = dict()
                for stream in activity_stream:  # Multiple types (e.g. distance, elevation, latlng)
                    if stream["type"] == "latlng":
                        data["latitude"] = [x[0] for x in stream["data"]]
                        data["longitude"] = [x[1] for x in stream["data"]]
                    elif stream["type"] == "altitude":
                        data["elevation"] = stream["data"]
                    else:
                        data[stream["type"]] = stream["data"]

                data["activity_id"] = stream["activity_id"]

                streams.append(pd.DataFrame(data))
                pbar.update(1)

        streams = pd.concat(streams)[[
            "activity_id", "time", "distance", "elevation", "latitude", "longitude"
        ]]

        if file:
            streams.to_pickle(file)

        return streams

    @staticmethod
    def create_gpx_files(activities: pd.DataFrame, activity_streams: pd.DataFrame, save_dir: str) -> None:
        """
        Create GPX files for all passed activity streams. Note that activities must be passed to
        get the start date

        :param activities: Pandas DataFrame containing the cleaned activity data
        :param activity_streams: Pandas DataFrame containing the cleaned activity streams
        :param save_dir: Directory to save the GPX files to
        :return: None
        """
        merged = activity_streams.merge(
            right=activities[["activity_id", "start_date_local"]],
            how="inner",
            on="activity_id"
        )

        merged["timestamp"] = merged["start_date_local"] + merged["time"].apply(lambda x: timedelta(seconds=x))
        merged = merged.drop(columns="start_date_local")

        # Create GPX file for each activity stream that has lat/long information
        with tqdm(total=len(merged["activity_id"].unique())) as pbar:
            for activity_id, stream in merged.groupby(by="activity_id"):
                pbar.set_description(f"Processing activity {activity_id}")

                if stream["latitude"].isna().any():
                    # If we have no location data, move on
                    pbar.update(1)
                    continue

                gpx = gpxpy.gpx.GPX()

                # Create track in our GPX file
                track = gpxpy.gpx.GPXTrack()
                gpx.tracks.append(track)

                # Create segment in our GPX track
                segment = gpxpy.gpx.GPXTrackSegment()
                track.segments.append(segment)

                # Create points
                for _, row in stream.iterrows():
                    segment.points.append(gpxpy.gpx.GPXTrackPoint(
                        latitude=row["latitude"],
                        longitude=row["longitude"],
                        elevation=row["elevation"],
                        time=row["timestamp"]
                    ))

                # Write GPX file
                with open(f"{save_dir}/{activity_id}.gpx", "w") as f:
                    f.write(gpx.to_xml())

                pbar.update(1)

        return merged
