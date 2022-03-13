from django.urls import path, include
import accounts.views

app_name = 'accounts'

urlpatterns = [
    path('login', accounts.views.login, name='login'),
    path('logout/', accounts.views.logout, name='logout'),
    path('registration/', accounts.views.registration, name='registration'),
]