from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.db import connection
from modules.forecast.predict import make_predict
from django.views.decorators.csrf import csrf_exempt


# from snippets.models import Snippet
# from snippets.serializers import SnippetSerializer

@csrf_exempt
@api_view(['GET', 'POST'])
def get_forecast(request):
    print("cast")
    if request.method == 'GET':
        result = make_predict(id_store=1, type=1)
        return Response({'name': result}, status=status.HTTP_201_CREATED)
    if request.method == 'POST':
        cursor = connection.cursor()
        print(request.data)
        type = request.data['type']
        store = request.data['store']

        result = make_predict(id_store=store, type=type)

        context = {
            result
        }
        return Response(context, status=status.HTTP_201_CREATED)
    return Response("something wrong", status=status.HTTP_400_BAD_REQUEST)
