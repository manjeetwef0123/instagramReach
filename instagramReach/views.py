from django.shortcuts import render
from joblib import load
import numpy as np
model=load('./savedModels/model.joblib')
# Create your views here.
def predictor(request):
    if request.method == 'POST':
        
        likes = int(request.POST.get('likes', 0))
        saves = int(request.POST.get('saves', 0))
        comments = int(request.POST.get('comments', 0))
        shares = int(request.POST.get('shares', 0))
        profile_visits = int(request.POST.get('profile_visits', 0))
        follows = int(request.POST.get('follows', 0))

        input_data=[likes, saves, comments, shares, profile_visits, follows]
        y_pred=np.round(model.predict([input_data]))
        print(y_pred)
        return render(request, 'main.html',{'result':y_pred})
    return render(request, 'main.html')


    
