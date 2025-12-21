import pandas as pd
import numpy as np


class GeolocationProcessor:
    """IP address to country mapping using numeric IP ranges."""

    def __init__(self, ip_df: pd.DataFrame):
        self.ip_df = ip_df.sort_values("lower_bound_ip_address")

    def ip_to_int(self, ip) -> int | None:
        """Convert IP to integer (dataset already stores numeric IPs)."""
        if pd.isna(ip):
            return None
        return int(ip)

    def map_country(self, ip_int: int) -> str:
        if ip_int is None:
            return "Unknown"

        row = self.ip_df[
            (self.ip_df["lower_bound_ip_address"] <= ip_int) &
            (self.ip_df["upper_bound_ip_address"] >= ip_int)
        ]

        return row.iloc[0]["country"] if not row.empty else "Unknown"
