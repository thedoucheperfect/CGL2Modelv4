# weather.py
import requests
import math

def get_wbt():
    """Fetch current weather data and calculate Wet Bulb Temperature (WBT)"""
    try:
        # Coordinates for JSW Kalmeshwar
        latitude = 21.2273
        longitude = 78.9002

        # Open-Meteo API URL
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m"
        
        # Fetch data with timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Validate response structure
        current = data.get('current', {})
        T = current.get('temperature_2m')
        RH = current.get('relative_humidity_2m')
        
        if None in (T, RH):
            raise ValueError("Missing weather data in API response")

        # Calculate WBT using Stull's formula
        WBT = (
            T * math.atan(0.151977 * math.sqrt(RH + 8.313659)) +
            math.atan(T + RH) -
            math.atan(RH - 1.676331) +
            0.00391838 * RH ** 1.5 * math.atan(0.023101 * RH) -
            4.686035
        )

        return WBT

    except Exception as e:
        print(f"Weather API error: {str(e)}")
        return None
