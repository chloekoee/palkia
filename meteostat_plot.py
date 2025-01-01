# Import Meteostat library and dependencies
from datetime import datetime
from meteostat import Hourly

# Set time period
start = datetime(2024, 12, 25, 23, 59)
end = datetime(2024, 12, 27, 23, 59)

# Get hourly data
data = Hourly("94695", start, end)
data = data.fetch()

# Print DataFrame
print(data)
# print(data)
