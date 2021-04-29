# Mecab 설치하고 사용하기



## 0. sudo명령어가 안먹힐 때

Mecab을 설치하기 위해서는 sudo 명령어를 사용해야 하고

혹시 sudo 명령어가 안 먹힐 때가 있는데,

이를 해결하기 위해서는 먼저 apt로 패키지를 설치할 수 있는 root 계정으로 접근해야 한다.

```bash
su
```

- 관리자 권한으로 로그인 하는 명령어
- 비밀번호를 물어보면 비밀번호를 입력해야 한다.



이 상황에서 sudo 패키지를 설치해주어야 한다.

```bash
apt-get install sudo
```







## 1. Install dependencies

```bash
# Install Java 1.8 or up
$ sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl

```



## 2. Install KoNLPy

```bash
$ python3 -m pip install --upgrade pip
$ python3 -m pip install konlpy       # Python 3.x

```



## 3. Install MeCab

```bash
$ sudo apt-get install curl git
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

