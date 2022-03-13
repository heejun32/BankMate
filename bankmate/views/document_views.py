from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone

from accounts.views import curr_user
from .. import face_recognition
from ..forms import DocumentForm
from ..models import Document


@login_required(login_url='accounts:login')
def document_create(request):
    # 결재 서류 등록
    if request.method == 'POST':
        user = curr_user.who_is_now()
        ID_NUM = User.objects.get(username=user)

        # 얼굴 인증
        face_recognition.FaceCon()

        if face_recognition.VGGFACE(ID_NUM).result == True:
            form = DocumentForm(request.POST)

            if form.is_valid():
                Document = form.save(commit=False)
                Document.author = request.user  # author 속성에 로그인 계정 저장
                Document.create_date = timezone.now()
                Document.save()
                return redirect('bankmate:index')

        else:
            return render(request, 'accounts/login.html', {'error' : '안면 인증 오류 및 사원번호 오류'})

    else:
        form = DocumentForm()
    context = {'form': form}
    return render(request, 'bankmate/document_form.html', context)

@login_required(login_url='accounts:login')
def document_modify(request, document_id):
    """
    bankmate 문서 수정
    """
    document = get_object_or_404(Document, pk=document_id)
    if request.user != document.author:
        messages.error(request, '수정권한이 없습니다')
        return redirect('bankmate:detail', document_id=document.id)

    if request.method == "POST":
        form = DocumentForm(request.POST, instance=document)
        if form.is_valid():
            document = form.save(commit=False)
            document.modify_date = timezone.now()  # 수정일시 저장
            document.save()
            return redirect('bankmate:detail', document_id=document.id)
    else:
        form = DocumentForm(instance=document)
    context = {'form': form}
    return render(request, 'bankmate/document_form.html', context)

@login_required(login_url='accounts:login')
def document_delete(request, document_id):
    """
    질문삭제
    """
    document = get_object_or_404(Document, pk=document_id)
    if request.user != document.author:
        messages.error(request, '삭제권한이 없습니다')
        return redirect('bankmate:detail', document_id=document.id)
    document.delete()
    return redirect('bankmate:index')