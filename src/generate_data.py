import numpy as np
import pandas as pd


def simulate_sensor_data(num_samples: int = 1000, num_features: int = 3) -> pd.DataFrame:
    t = np.arange(num_samples)
    data = {
        f'sensor_{i+1}': np.sin(0.02 * t + i) + np.random.normal(scale=0.1, size=num_samples)
        for i in range(num_features)
    }
    # Add simple anomalies
    if num_samples > 200:
        data['sensor_1'][200:220] += 3
    return pd.DataFrame(data)


def main():
    df = simulate_sensor_data()
    raw_path = 'data/raw'
    import os
    os.makedirs(raw_path, exist_ok=True)
    df.to_csv(f'{raw_path}/sensor_data.csv', index=False)


if __name__ == '__main__':
    main()
