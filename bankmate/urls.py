from django.urls import path, include

import chatbot.views
from .views import base_views, document_views, answer_views, comment_views
app_name = 'bankmate'

urlpatterns = [
    path('', base_views.index, name='index'),
    path('first', base_views.first, name='first'),
    path('main', base_views.main, name='main'),
    path('answerChat', base_views.answerChat, name='answerChat'),

    path('<int:document_id>/',
         base_views.detail, name='detail'),

    # question_views.py
    path('document/create/',
         document_views.document_create, name='document_create'),
    path('document/modify/<int:document_id>/',
         document_views.document_modify, name='document_modify'),
    path('document/delete/<int:document_id>/',
         document_views.document_delete, name='document_delete'),

    # answer_views.py
    path('answer/create/<int:document_id>/',
         answer_views.answer_create, name='answer_create'),
    path('answer/modify/<int:answer_id>/',
         answer_views.answer_modify, name='answer_modify'),
    path('answer/delete/<int:answer_id>/',
         answer_views.answer_delete, name='answer_delete'),

    # comment_views.py
    path('comment/create/document/<int:document_id>/',
         comment_views.comment_create_document, name='comment_create_document'),
    path('comment/modify/document/<int:comment_id>/',
         comment_views.comment_modify_document, name='comment_modify_document'),
    path('comment/delete/document/<int:comment_id>/',
         comment_views.comment_delete_document, name='comment_delete_document'),
    path('comment/create/answer/<int:answer_id>/',
         comment_views.comment_create_answer, name='comment_create_answer'),
    path('comment/modify/answer/<int:comment_id>/',
         comment_views.comment_modify_answer, name='comment_modify_answer'),
    path('comment/delete/answer/<int:comment_id>/',
         comment_views.comment_delete_answer, name='comment_delete_answer'),

]