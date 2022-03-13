from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth
from bankmate import face_recognition, face_registration
from .forms import CurrentUser
import os

curr_user = CurrentUser()

def registration(request):
    # 사원번호 입력과 함께 얼글 등록 진행

        if request.method == 'POST' and request.POST['사원번호']:
            ID_NUM = request.POST['사원번호']
            database = os.listdir("E:\Github\Bankmate\\database")

            try:
                ID_NUM = int(ID_NUM)
            except(ValueError):
                return render(request, 'accounts/registration.html', {'error': '올바른 사원번호를 입력하세요.'})

            if str(ID_NUM) +'.jpg' in database:
                return render(request, 'accounts/login.html', {'error': '등록되어 있는 사원입니다. 로그인을 해주세요.'})

            face_registration.FaceReg(request.POST['사원번호'])                     # 사원번호 등록 후 얼굴 등록 시작
            user = User.objects.create_user(username=request.POST['사원번호'])      # 사원 객채 생성 및 로그인 (시간이 된다면 아이디 중복체크 해주기)
            curr_user.set_current_user(int(request.POST['사원번호']))               # 현재 유저 정보 가져오기
            auth.login(request, user)
            return redirect('/')
        else:
            return render(request, 'accounts/registration.html')


def login(request):
    if request.method == 'POST' and request.POST['사원번호']:    # 사원번호를 작성하고 요청시
        ID_NUM = request.POST['사원번호']
        database = os.listdir("E:\Github\Bankmate\\database")

        if str(ID_NUM) +'.jpg'  not in database:
            return render(request, 'accounts/registration.html', {'error': '존재하지 않는 사원입니다. 사원 등록을 진행해주세요.'})

        face_recognition.FaceCon()  # 얼굴 인식 시작

        if face_recognition.VGGFACE(ID_NUM).result == True:
            user = User.objects.get(username=ID_NUM)
            curr_user.set_current_user(int(ID_NUM))         # 현재 유저 정보 가져오기
            auth.login(request, user)
            return redirect('/bankmate')
        else:
            return render(request, 'accounts/login.html', {'error' : '얼굴이 일치하지 않습니다. 다시 시도해 주세요.'})
        
    elif request.method == 'POST':
        return render(request, 'accounts/login.html', {'error' : '사원 번호를 입력해주세요.'})
    else:
        return render(request, 'accounts/login.html')

def logout(request):
    if request.method == 'GET':
        curr_user.init_current_user()
        auth.logout(request)
        return redirect('/first')