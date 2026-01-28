import simpy
import random
import numpy as np

def queue_simulation(arrival_rate, service_rate, servers, max_customers):
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=servers)
    wait_times = []

    def customer(env):
        arrive = env.now
        with server.request() as req:
            yield req
            wait_times.append(env.now - arrive)
            service_time = random.expovariate(1 / service_rate)
            yield env.timeout(service_time)

    def arrival_process(env):
        for _ in range(max_customers):
            env.process(customer(env))
            yield env.timeout(random.expovariate(1 / arrival_rate))

    env.process(arrival_process(env))
    env.run()

    return np.mean(wait_times)
