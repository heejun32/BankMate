from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect, resolve_url
from django.utils import timezone

from ..forms import CommentForm
from ..models import Document, Answer, Comment

@login_required(login_url='accounts:login')
def comment_create_document(request, document_id):
    """
    질문댓글등록
    """
    document = get_object_or_404(Document, pk=document_id)
    if request.method == "POST":
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.author = request.user
            comment.create_date = timezone.now()
            comment.document = document
            comment.save()
            return redirect('{}#comment_{}'.format(
                resolve_url('bankmate:detail', document_id=comment.document.id), comment.id))
    else:
        form = CommentForm()
    context = {'form': form}
    return render(request, 'bankmate/comment_form.html', context)

@login_required(login_url='accounts:login')
def comment_modify_document(request, comment_id):
    """
    질문댓글수정
    """
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, '댓글수정권한이 없습니다')
        return redirect('bankmate:detail', document_id=comment.document.id)

    if request.method == "POST":
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.modify_date = timezone.now()
            comment.save()
            return redirect('{}#comment_{}'.format(
                resolve_url('bankmate:detail', document_id=comment.document.id), comment.id))
    else:
        form = CommentForm(instance=comment)
    context = {'form': form}
    return render(request, 'bankmate/comment_form.html', context)

@login_required(login_url='accounts:login')
def comment_delete_document(request, comment_id):
    """
    질문댓글삭제
    """
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, '댓글삭제권한이 없습니다')
        return redirect('bankmate:detail', document_id=comment.document.id)
    else:
        comment.delete()
    return redirect('bankmate:detail', document_id=comment.document.id)

@login_required(login_url='accounts:login')
def comment_create_answer(request, answer_id):
    """
    답글댓글등록
    """
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.method == "POST":
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.author = request.user
            comment.create_date = timezone.now()
            comment.answer = answer
            comment.save()
            return redirect('{}#comment_{}'.format(
                resolve_url('bankmate:detail', document_id=comment.answer.document.id), comment.id))
    else:
        form = CommentForm()
    context = {'form': form}
    return render(request, 'bankmate/comment_form.html', context)


@login_required(login_url='accounts:login')
def comment_modify_answer(request, comment_id):
    """
    답글댓글수정
    """
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, '댓글수정권한이 없습니다')
        return redirect('bankmate:detail', document_id=comment.answer.document.id)

    if request.method == "POST":
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.modify_date = timezone.now()
            comment.save()
            return redirect('{}#comment_{}'.format(
                resolve_url('bankmate:detail', document_id=comment.answer.document.id), comment.id))
    else:
        form = CommentForm(instance=comment)
    context = {'form': form}
    return render(request, 'bankmate/comment_form.html', context)


@login_required(login_url='accounts:login')
def comment_delete_answer(request, comment_id):
    """
    답글댓글삭제
    """
    comment = get_object_or_404(Comment, pk=comment_id)
    if request.user != comment.author:
        messages.error(request, '댓글삭제권한이 없습니다')
        return redirect('bankmate:detail', document_id=comment.answer.document.id)
    else:
        comment.delete()
    return redirect('bankmate:detail', document_id=comment.answer.document.id)