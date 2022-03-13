from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect, resolve_url
from django.utils import timezone

from ..forms import AnswerForm
from ..models import Document, Answer

@login_required(login_url='accounts:login')
def answer_create(request, document_id):
    document = get_object_or_404(Document, pk=document_id)

    if request.method == "POST":
        form = AnswerForm(request.POST)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.author = request.user
            answer.create_date = timezone.now()
            answer.document = document
            answer.save()
            return redirect('{}#answer_{}'.format(
                resolve_url('bankmate:detail', document_id=document.id), answer.id))
    else:
        form = AnswerForm()
    context = {'document': document, 'form': form}
    return render(request, 'bankmate/document_detail.html', context)

@login_required(login_url='accounts:login')
def answer_modify(request, answer_id):
    """
    답변 수정
    """
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('bankmate:detail', document_id=answer.document.id)

    if request.method == "POST":
        form = AnswerForm(request.POST, instance=answer)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.modify_date = timezone.now()
            answer.save()
            return redirect('{}#answer_{}'.format(
                resolve_url('bankmate:detail', document_id=answer.document.id), answer.id))
    else:
        form = AnswerForm(instance=answer)
    context = {'answer': answer, 'form': form}
    return render(request, 'bankmate/answer_form.html', context)

@login_required(login_url='accounts:login')
def answer_delete(request, answer_id):
    """
    답변삭제
    """
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, '삭제권한이 없습니다')
    else:
        answer.delete()
    return redirect('bankmate:detail', document_id=answer.document.id)