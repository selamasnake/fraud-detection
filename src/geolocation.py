import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@dataclass
class GeoConfig:
    """
    Configuration for GeolocationProcessor.

    Attributes:
        unknown_label (str): Label to return when IP cannot be mapped to a country.
    """
    unknown_label: str = "Unknown"

class GeolocationProcessor:
    """
    Map numeric IP addresses to countries using an IP range DataFrame.

    Design decisions:
        - Uses IntervalIndex for efficient single-IP lookups.
        - Configurable unknown label.
        - Supports vectorized mapping of pandas Series.
    """

    def __init__(self, ip_df: pd.DataFrame, config: GeoConfig | None = None):
        """
        Initialize the GeolocationProcessor.

        Args:
            ip_df (pd.DataFrame): DataFrame containing IP ranges and countries.
                Must have columns:
                    - 'lower_bound_ip_address'
                    - 'upper_bound_ip_address'
                    - 'country'
            config (GeoConfig | None): Optional configuration. Defaults to GeoConfig().
        """
        self.ip_df = ip_df.sort_values("lower_bound_ip_address")
        self.config = config or GeoConfig()
        # Create IntervalIndex for fast IP lookups
        self.intervals = pd.IntervalIndex.from_arrays(
            self.ip_df["lower_bound_ip_address"],
            self.ip_df["upper_bound_ip_address"],
            closed="both"
        )

    def ip_to_int(self, ip: str | int | float | None) -> int | None:
        """
        Convert an IP to integer.

        Args:
            ip (str | int | float | None): IP address to convert.

        Returns:
            int | None: Integer representation of IP, or None if input is NaN.
        """
        if pd.isna(ip):
            return None
        return int(ip)

    def map_country(self, ip_int: int | None) -> str:
        """
        Map a single numeric IP to its corresponding country.

        Args:
            ip_int (int | None): Numeric IP address to map.

        Returns:
            str: Country name or the configured unknown_label if not found.
        """
        if ip_int is None:
            return self.config.unknown_label

        # Lookup using IntervalIndex
        idx = self.intervals.get_indexer([ip_int])[0]
        if idx == -1:
            logging.debug(f"IP {ip_int} not in any range, returning '{self.config.unknown_label}'")
            return self.config.unknown_label
        return self.ip_df.iloc[idx]["country"]
    
    def map_countries_vectorized(self, ip_series: pd.Series) -> pd.Series:
        """
        Map a pandas Series of numeric IPs to countries.

        Args:
            ip_series (pd.Series): Series containing IP addresses to map.

        Returns:
            pd.Series: Series of country names corresponding to each IP.
        """
        ip_ints = ip_series.apply(self.ip_to_int)
        return ip_ints.apply(self.map_country)
