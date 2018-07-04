from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.db import connection
from modules.forecast.predict import forecast
from django.views.decorators.csrf import csrf_exempt


# from snippets.models import Snippet
# from snippets.serializers import SnippetSerializer

@csrf_exempt
@api_view(['GET', 'POST'])
def get_forecast(request):
    print("cast")
    if request.method == 'GET':
        result = forecast(100, name='revenue_1_1')
        return Response({'name': result}, status=status.HTTP_201_CREATED)
    if request.method == 'POST':
        cursor = connection.cursor()
        print(request.data)
        object = request.data['object']
        name = request.data['name']

        result = forecast(object, name='revenue_1_1')

        context = {
            result
        }
        return Response(context, status=status.HTTP_201_CREATED)
    return Response("something wrong", status=status.HTTP_400_BAD_REQUEST)
