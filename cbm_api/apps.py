from django.apps import AppConfig
from django.conf import settings
# from energy_consumption_ml.ml_code.energy import RegressorSwitcher
import joblib


class CbmApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cbm_api'
    model_file = settings.MODELS/'decision_tree.joblib'
    predictor = joblib.load(model_file)


# print(EnergyConsumptionApiConfig.model)
