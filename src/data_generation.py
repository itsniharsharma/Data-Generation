import random
import pandas as pd
from simulator import queue_simulation

def generate_dataset(n=1000):
    data = []

    for _ in range(n):
        arrival_rate = random.uniform(1, 5)
        service_rate = random.uniform(1, 6)
        servers = random.randint(1, 5)
        max_customers = random.randint(50, 200)

        avg_wait = queue_simulation(
            arrival_rate,
            service_rate,
            servers,
            max_customers
        )

        data.append([
            arrival_rate,
            service_rate,
            servers,
            max_customers,
            avg_wait
        ])

    df = pd.DataFrame(data, columns=[
        "arrival_rate",
        "service_rate",
        "servers",
        "max_customers",
        "avg_wait_time"
    ])

    df.to_csv("data/simpy_dataset.csv", index=False)
    return df

if __name__ == "__main__":
    generate_dataset()
