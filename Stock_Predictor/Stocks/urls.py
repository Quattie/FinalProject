from django.urls import path
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('', views.home, name='stock-home'),
    path('history', views.history, name='stock-history'),
    path('about/', views.about, name='stock-about'),
    path('recurrent/', views.recurrent, name='stock-recurrent'),
    path('svm/', views.svm, name='stock-svm'),
    path('random-forests/', views.randomForest, name='stock-random-forests'),
    path('random-regressor/', views.randomForestRegressor,
         name='stock-random-regressor'),
    path('crypto/', views.crypto, name='stock-crypto'),
    path('crypto-model/', views.cryptoModel, name='stock-crypto-model')
]

urlpatterns += staticfiles_urlpatterns()
