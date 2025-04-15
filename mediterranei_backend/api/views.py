from rest_framework.response import Response
from rest_framework.decorators import api_view
from .services.llm import load_model

@api_view(['POST'])
def get_review_analysis(request):
    model = load_model()
    moderation_verdict,  moderation_verdict_confidence = model.classify_moderation(request.POST.get('review'))
    data = {
        "echo": request.POST.get("review"),
        "spam": model.isSpam(request.POST.get('review')),
        "needs_moderation": {
            "verdict": moderation_verdict,
            "confidence": moderation_verdict_confidence,
            },
    }
    return Response(data)