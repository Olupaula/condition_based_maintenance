from django.shortcuts import render
from rest_framework.views import APIView, Response
from .apps import CbmApiConfig


class PredictDegradationStateApi(APIView):
    def post(self, request):
        request = request.data

        gas_turbine_shaft_torgue = request['Gas Turbine shaft torque (GTT) [kN m]']
        gas_generator_rate_of_revolutions = request['Gas Generator rate of revolutions (GGn) [rpm]']

        predictor = CbmApiConfig.predictor
        predict_degradation_state = predictor.predict([[
            gas_turbine_shaft_torgue,
            gas_generator_rate_of_revolutions
        ]])

        response_dict = {'performance_degradation_state': predict_degradation_state[0]}

        return Response(response_dict, status=200)

    def get(self, request):
        context = {}
        return render(request, 'api_home.html', context)


cbm_api = PredictDegradationStateApi.as_view()

