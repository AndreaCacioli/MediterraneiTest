from rest_framework.response import Response
from rest_framework.decorators import api_view

@api_view(['POST'])
def getReview(request):
    print(request)
    data = {
        "echo": request.POST.get("review")
    }
    return Response(data)