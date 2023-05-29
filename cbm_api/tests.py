from django.test import TestCase
import requests

endpoint = "http://127.0.0.1:8000/cbm_api"
data_dict = {
            'Gas Turbine shaft torque (GTT) [kN m]': 12,
            'Gas Generator rate of revolutions (GGn) [rpm]': 2
}

predict = requests.post(endpoint, json=data_dict)

performance_degradation_state_dict = predict.json()
print("Performance Degradation State Dictionary = ",  performance_degradation_state_dict)

performance_degradation_state = performance_degradation_state_dict['performance_degradation_state']
print("Performance Degradation State = ", performance_degradation_state)
