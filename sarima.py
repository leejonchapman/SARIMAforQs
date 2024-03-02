import numpy as np
import pandas as pd
from pmdarima import auto_arima

# Historical data for Qs

quarters = np.array([i for i in range(1, 19)])
totals = np.array([1, 32, 45, 56, 678, 6887, 666, 5675, 8888, 9999,
                   12121, 12124, 14241, 15151, 16161, 17171, 19191, 200000])

print("Length of 'quarters':", len(quarters))
print("Length of 'totals':", len(totals))


# Check if the lengths are the same
if len(quarters) != len(totals):
    raise ValueError("The lengths of 'quarters' and 'totals' must be the same.")

# Create a pandas DataFrame
data = pd.DataFrame({'Quarters': quarters, 'Totals': totals})
# Fit SARIMA model
sarima_model = auto_arima(data['Totals'], seasonal=True, m=4, trace=True, suppress_warnings=True)

# Forecast the next 4 quarters
forecast_values, conf_int = sarima_model.predict(n_periods=4, return_conf_int=True)

# Print the forecasted values
for i, prediction in enumerate(forecast_values):
    print(f'Q{quarters[-1] + i + 1}: £{prediction:.2f}')
