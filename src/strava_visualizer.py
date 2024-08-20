import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL
import requests
import time

from io import BytesIO
from tqdm.auto import tqdm


class StravaVisualizer:
    MAX_SIZE = (2160, 3840)  # 4k - only used for auto zoom
    MARGIN_SIZE = 32  # Pixel margin around track points

    # OPEN_STREET_MAPS_TILE_SERVER = "https://maps.wikimedia.org/osm-intl/{}/{}/{}.png"  # Fights you to download
    # OPEN_STREET_MAPS_TILE_SERVER = "https://tile.openstreetmap.org/{}/{}/{}.png"
    # OPEN_STREET_MAPS_TILE_SERVER = "https://a.tile-cyclosm.openstreetmap.fr/cyclosm/{}/{}/{}.png"
    OPEN_STREET_MAPS_TILE_SERVER = "https://maps.geoapify.com/v1/tile/osm-bright-grey/{}/{}/{}.png"
    OPEN_STREET_MAPS_TILE_SIZE = 256
    OPEN_STREET_MAPS_MAX_ZOOM = 19
    OPEN_STREET_MAPS_MAX_TILE_COUNT = 1500

    def __init__(self, api_key: str = None, override_max_zoom: bool = False) -> None:
        self._api_key = api_key
        self.bad_urls = []
        self._OVERRIDE_MAX_ZOOM = override_max_zoom

    @staticmethod
    def lat_long_to_osm(lat: float, lon: float, zoom: int) -> tuple[float, float]:
        """
        Returns open street maps coordinates (x, y) from (lat, long)
        See https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

        :param lat: Latitude in degrees
        :param lon: Longitude in degrees
        :param zoom: Zoom level
        :return: Tuple containing cartesian open street maps coordinates (x, y)
        """
        n = 2 ** zoom
        lat = np.radians(lat)
        x = (lon + 180.00) / 360.0 * n
        y = (1.0 - np.arcsinh(np.tan(lat)) / np.pi) / 2.0 * n

        return x, y

    @staticmethod
    def osm_to_lat_long(x: float, y: float, zoom: int) -> tuple[float, float]:
        """
        Returns (lat, lon) in degrees from open street maps coordinates x, y

        :param x: Open street map coordinate x
        :param y: Open street map coordinate y
        :param zoom: Zoom level
        :return: Tuple containing latitude, longitude in degrees
        """

        n = 2.0 ** zoom
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n))))
        lon = x / n * 360.0 - 180.0

        return lat, lon

    @staticmethod
    def apply_gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply a gaussian filter with standard deviation sigma to a passed image

        :param image: NumPy array to apply the filter to
        :param sigma: Standard deviation to use in the filtering
        :return: NumPy array representation of the filtered image
        """
        i, j = np.meshgrid(
            np.arange(image.shape[0]),
            np.arange(image.shape[1]),
            indexing="ij"  # Matrix style indexing
        )

        mu = (image.shape[0] // 2, image.shape[1] // 2)
        distribution = 1.0 / (2.0 * np.pi * sigma ** 2) * np.exp(
            -0.5 * (((i - mu[0]) / sigma) ** 2 + ((j - mu[1]) / sigma) ** 2)
        )
        distribution = np.roll(distribution, (-mu[0], -mu[1]), axis=(0, 1))

        # FFT image/gaussian
        image_fft = np.fft.rfft2(image)
        distribution_fft = np.fft.rfft2(distribution)

        return np.fft.irfft2(image_fft * distribution_fft)

    def download_tile(self, tile_url: str, tile_file: str) -> bool:
        """
        Download tile from url to file. Return True if download was successful

        :param tile_url: URL from which to download the tile
        :param tile_file: File to save the tile to
        :return: Boolean indicating whether the download was successful
        """
        if self._api_key:
            response = requests.get(
                url=tile_url,
                params={"apiKey": self._api_key},
                headers={"User-Agent": "StravaVizBot/1.0 (https://github.com/zsnyder21/StravaViz; zsnyder21@gmail.com)"}
            )
        else:
            response = requests.get(
                url=tile_url,
                headers={"User-Agent": "StravaVizBot/1.0 (https://github.com/zsnyder21/StravaViz; zsnyder21@gmail.com)"}
            )

        try:
            tile = PIL.Image.open(BytesIO(response.content))
            tile.save(tile_file)
        except PIL.UnidentifiedImageError:
            return False

        # Sleep for 1-2 seconds to rate-limit ourselves
        if not self._api_key:
            rng = np.random.default_rng()
            time.sleep(1 + rng.uniform())

        return True

    @staticmethod
    def _process_gpx_files(
            gpx_dir: str,
            date_min: str = "1900-01-01",
            date_max: str = "2100-12-31",
            lat_lon_bounds: tuple[float, float, float, float] = (-90.0, 90.0, -180.0, 180.0)
    ) -> tuple[np.ndarray, int]:
        """
        Iterates through the GPX files in the passed directory and extracts the lat/lon data from them

        :param gpx_dir: Directory containing the GPX files
        :param date_min: Use only GPX tracks with dates after this date
        :param date_max: Use only GPX tracks with dates before this date
        :param lat_lon_bounds: The latitude/longitude boundaries to use when filtering the GPX files
        :return: NumPy array containing the lat/lon data and an integer number of tracks used
        """
        date_min, date_max = pd.to_datetime(date_min), pd.to_datetime(date_max)
        gpx_files = glob.glob(f"{gpx_dir}/*.gpx")
        lat_bound_min, lon_bound_min, lat_bound_max, lon_bound_max = lat_lon_bounds

        if not gpx_files:
            raise ValueError(f"Error: No GPX files found within {gpx_dir}")

        lat_lon_data = []
        track_count = 0
        with tqdm(total=len(gpx_files)) as pbar:
            for gpx_file in gpx_files:
                pbar.set_description(f"Processing {os.path.basename(gpx_file)}")

                # Determine the year the activity started
                with open(gpx_file, encoding="utf-8") as f:
                    for line in f.readlines():
                        if "<time>" in line:
                            date = pd.to_datetime(line.split(">")[1][:11])
                            break

                with open(gpx_file, encoding="utf-8") as f:
                    counted = False
                    for line in f.readlines():
                        # if "<time" in line:
                        #     year = line.split(">")[1][:4]
                        if date_min <= date <= date_max:
                            if "<trkpt" in line:
                                lat_lon = line.split('"')

                                lat, lon = float(lat_lon[1]), float(lat_lon[3])
                                if (lat_bound_min < lat < lat_bound_max) and (lon_bound_min < lon < lon_bound_max):
                                    lat_lon_data.append([lat, lon])
                                    track_count += not counted
                                    counted = True

                        else:
                            continue

                pbar.update(1)

        lat_lon_data = np.array(lat_lon_data)

        if lat_lon_data.size == 0:
            raise ValueError(f"Error: No lat/lon data found matching filter criteria")

        return lat_lon_data, track_count

    def _enumerate_tiles(self, lat_lon_bounds: np.ndarray, zoom: int) -> tuple[int, int, int, int, int]:
        """
        Auto zoom if necessary, and enumerate which map tiles we will need

        :param lat_lon_data: NumPy array containing the lat/lon data
        :param zoom: Zoom level specified
        :return: Tuple containing zoom, x_min, x_max, y_min, y_max
        """
        # Find tile coordinates
        lat_min, lon_min, lat_max, lon_max = lat_lon_bounds
        # lat_min, lon_min = np.min(lat_lon_data, axis=0)  # Auto crop to GPX data
        # lat_max, lon_max = np.max(lat_lon_data, axis=0)

        # Handle auto zoom
        if zoom > -1:
            zoom = min(zoom, self.OPEN_STREET_MAPS_MAX_ZOOM)

            x_min, y_max = map(int, self.lat_long_to_osm(lat_min, lon_min, zoom))
            x_max, y_min = map(int, self.lat_long_to_osm(lat_max, lon_max, zoom))

        else:
            # Auto zoom
            zoom = self.OPEN_STREET_MAPS_MAX_ZOOM

            while True:
                # Iterate from max zoom down until we fit inside MAX_SIZE
                x_min, y_max = map(int, self.lat_long_to_osm(lat_min, lon_min, zoom))
                x_max, y_min = map(int, self.lat_long_to_osm(lat_max, lon_max, zoom))

                if ((x_max - x_min + 1) * self.OPEN_STREET_MAPS_TILE_SIZE <= self.MAX_SIZE[0] and
                        (y_max - y_min + 1) * self.OPEN_STREET_MAPS_TILE_SIZE <= self.MAX_SIZE[1]):
                    break

                zoom -= 1
            print(f"Auto zoom set to {zoom}")

        return zoom, x_min, x_max, y_min, y_max

    def _construct_base_map(self, zoom: int, x_min: int, x_max: int, y_min: int, y_max: int) -> np.ndarray:
        """
        Download and stitch together the map tiles

        :param zoom: Zoom level
        :param x_min: Minimum x
        :param x_max: Maximum x
        :param y_min: Minimum y
        :param y_max: Maximum y
        :return: NumPy array containing the base map to use
        """
        tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
        if not self._OVERRIDE_MAX_ZOOM and tile_count > self.OPEN_STREET_MAPS_MAX_TILE_COUNT:
            raise ValueError(f"Error: Too many tiles ({tile_count}) would need to be downloaded")

        base_map = np.zeros((
            (y_max - y_min + 1) * self.OPEN_STREET_MAPS_TILE_SIZE,
            (x_max - x_min + 1) * self.OPEN_STREET_MAPS_TILE_SIZE,
            3
        ))

        # Download and stitch
        with tqdm(total=tile_count) as pbar:
            pbar.set_description(f"Downloading tiles")
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    tile_file = "../data/tiles/zoom_{}/tile_{}_{}_{}.png".format(zoom, zoom, x, y)
                    os.makedirs(os.path.dirname(tile_file), exist_ok=True)

                    if not glob.glob(tile_file):
                        tile_url = self.OPEN_STREET_MAPS_TILE_SERVER.format(zoom, x, y)

                        if not self.download_tile(tile_url=tile_url, tile_file=tile_file):
                            print(f"Error downloading tile {tile_url}, using blank tile")
                            self.bad_urls.append(tile_url)
                            tile = np.ones((self.OPEN_STREET_MAPS_TILE_SIZE, self.OPEN_STREET_MAPS_TILE_SIZE, 3))
                            plt.imsave(tile_file, tile)

                    pbar.update(1)

                    tile = plt.imread(tile_file)
                    i = y - y_min
                    j = x - x_min

                    base_map[
                        i * self.OPEN_STREET_MAPS_TILE_SIZE:(i + 1) * self.OPEN_STREET_MAPS_TILE_SIZE,
                        j * self.OPEN_STREET_MAPS_TILE_SIZE:(j + 1) * self.OPEN_STREET_MAPS_TILE_SIZE,
                        :
                    ] = tile[:, :, :3]

        base_map = np.sum(base_map * [0.2126, 0.7152, 0.0722], axis=2)  # Greyscale
        base_map = 1.0 - base_map  # Invert colors
        base_map = np.dstack((base_map, base_map, base_map))  # RGB

        return base_map

    def _overlay_tracks(
            self,
            base_map: np.ndarray,
            lat_lon_data: np.ndarray,
            zoom: int,
            x_min: int,
            y_min: int,
            sigma: int,
            brightness_factor: float,
            track_count: int
    ) -> np.ndarray:
        """
        Overaly the GPS track on top of the passed map

        :param base_map: The map to overlay the tracks on
        :param lat_lon_data: The tracks
        :param zoom: Zoom level
        :param x_min: Minimum x
        :param y_min: Minimum y
        :param sigma: Pixel width of the tracks
        :param brightness_factor: Manually adjust brightness
        :param track_count: Number of tracks used
        :return: NumPy array containing the overlayed tracks on the passed map
        """
        sigma_pixel = sigma
        data = np.zeros(base_map.shape[:2])
        xy_data = self.lat_long_to_osm(lat_lon_data[:, 0], lat_lon_data[:, 1], zoom)
        xy_data = np.array(xy_data).T
        xy_data = np.round((xy_data - [x_min, y_min]) * self.OPEN_STREET_MAPS_TILE_SIZE)
        ij_data = np.flip(xy_data.astype(int), axis=1)  # Convert to base_map coords

        for i, j in ij_data:
            data[i - sigma_pixel:i + sigma_pixel, j - sigma_pixel:j + sigma_pixel] += 1.0

        # Account for maximum accumulation of a track point
        # See https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        res_pixel = 156543.03 * np.cos(np.radians(np.mean(lat_lon_data[:, 0]))) / (2 ** zoom) / brightness_factor

        # Trackpoint max accumulation per pixel = 1/5 (trackpoint/meter) * res_pixel (meter/pixel) * activities
        m = max(1.0, np.round((1.0 / 5.0) * res_pixel * track_count))
        data[data > m] = m

        # Equalize histogram and compute kernel density estimation
        data_hist, _ = np.histogram(data, bins=int(m + 1))
        data_hist = np.cumsum(data_hist) / data.size  # cumsum normalized

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = m * data_hist[int(data[i, j])]

        data = self.apply_gaussian_filter(data, float(sigma_pixel))
        data = (data - data.min()) / (data.max() - data.min())  # Norm to [0, 1]

        # Color
        cmap = plt.get_cmap("hot")
        data_color = cmap(data)
        data_color[data_color == cmap(0.0)] = 0.0  # Remove background colors

        for c in range(3):
            base_map[:, :, c] = (1.0 - data_color[:, :, c]) * base_map[:, :, c] + data_color[:, :, c]

        # Crop
        # i_min, j_min = np.min(ij_data, axis=0)
        # i_max, j_max = np.max(ij_data, axis=0)

        # base_map = base_map[
        #            max(i_min - self.MARGIN_SIZE, 0):min(i_max + self.MARGIN_SIZE, base_map.shape[0]),
        #            max(j_min - self.MARGIN_SIZE, 0):min(j_max + self.MARGIN_SIZE, base_map.shape[1])
        #            ]

        return base_map

    def generate_heatmap(
            self,
            gpx_dir: str,
            file: str,
            zoom: int = -1,
            sigma: int = 1,
            date_min: str = "1900-01-01",
            date_max: str = "2100-12-31",
            brightness_factor: float = 1.0,
            lat_lon_bounds: tuple[float, float, float, float] = (-90.0, -180.0, 90.0, 180.0)
    ) -> np.ndarray:
        """
        Generate a heatmap using GPX files in a specified directory

        :param gpx_dir: Directory containing the gpx files (consumes all .gpx files)
        :param file: Location to save the heatmap to
        :param zoom: Zoom level to use when fetching map tiles
        :param sigma: Width of the strava tracks
        :param date_min: Use only GPX tracks with dates after this date
        :param date_max: Use only GPX tracks with dates before this date
        :param brightness_factor: Manually increase/decrease brightness
        :param lat_lon_bounds: Bounding latitudes/longitudes describing the map
        :return: NumPy array representation of the heatmap image
        """
        lat_lon_data, track_count = self._process_gpx_files(
            gpx_dir=gpx_dir,
            date_min=date_min,
            date_max=date_max,
            lat_lon_bounds=lat_lon_bounds
        )

        zoom, x_min, x_max, y_min, y_max = self._enumerate_tiles(lat_lon_bounds=lat_lon_bounds, zoom=zoom)
        base_map = self._construct_base_map(zoom=zoom, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        heatmap = self._overlay_tracks(
            base_map=base_map,
            lat_lon_data=lat_lon_data,
            zoom=zoom,
            x_min=x_min,
            y_min=y_min,
            sigma=sigma,
            brightness_factor=brightness_factor,
            track_count=track_count
        )

        # Save
        plt.imsave(file, heatmap)

        return heatmap


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    vis = StravaVisualizer(api_key=os.getenv("GEOAPIFY_API_KEY"))

    vis.generate_heatmap(
        gpx_dir="../data/gpx/",
        file="../img/heatmap_example.png",
        zoom=15,
        sigma=1,
        lat_lon_bounds=(39.813811, -105.558014, 40.166281, -105.195465)  # Boulder
        # lat_lon_bounds=(37.788624, -122.392159, 37.895718, -122.219810)  # Oakland
    )

    for url in vis.bad_urls:
        print(url)

