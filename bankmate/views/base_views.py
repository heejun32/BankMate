from django.core.paginator import Paginator
from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import Q
from ..models import Document
from django.views.decorators.csrf import csrf_exempt
from chatbot import Cnn

def main(request):
    return render(request, "bankmate/main.html")

def first(request):
    return render(request, "accounts/login.html")

def index(request):
    # 입력 파라미터
    page = request.GET.get('page', '1')  # 페이지
    kw = request.GET.get('kw', '')  # 검색어

    document_list = Document.objects.order_by('-create_date')
    if kw:
        document_list = document_list.filter(
            Q(subject__icontains=kw) |  # 제목검색
            Q(content__icontains=kw) |  # 내용검색
            Q(author__username__icontains=kw) |  # 질문 글쓴이검색
            Q(answer__author__username__icontains=kw)  # 답변 글쓴이검색
        ).distinct()

    # 페이징처리
    paginator = Paginator(document_list, 10)  # 페이지당 10개씩 보여주기
    page_obj = paginator.get_page(page)
    context = {'document_list': page_obj, 'page': page, 'kw': kw}
    return render(request, 'bankmate/document_list.html', context)


def detail(request, document_id):
    document = get_object_or_404(Document, pk=document_id)
    Document.objects.get(id=document_id)
    context = {'document': document}
    return render(request, 'bankmate/document_detail.html', context)


@csrf_exempt
def answerChat(request):
    # 사용자가 보낸 메세지를 views.py로 불러옴
    from django.http import JsonResponse
    input_keyword = request.POST.get("keyword")

    # 사용자가 보낸 메세지 확인
    print(input_keyword + "라는 키워드가 수신 됨")

    input_keyword1 = "\'" + input_keyword + "\'"
    res = str(Cnn.CnnBased(input_keyword1).print_answer())

    return JsonResponse(res, safe=False, json_dumps_params={'ensure_ascii': False}, status=200)