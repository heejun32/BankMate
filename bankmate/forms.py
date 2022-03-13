from django import forms
from bankmate.models import Document, Answer, Comment


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document  # 사용할 모델
        fields = ['subject', 'content']  # DocumentForm에서 사용할 Document 모델의 속성

        labels = {
            'subject': '제목',
            'content': '내용',
        }

class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content': '답변내용',
        }

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        labels = {
            'content': '댓글내용',
        }