## Google Colab Runtime 끊김 해결하기



> 출처 : https://stackoverflow.com/questions/57113226/how-to-prevent-google-colab-from-disconnecting
>
> **먼저, Colab Pro가 아니라면 하루 12시간 이상은 돌릴 수 없다.**



1. 실행시키고자 하는 노트북을 키고 **`F12`**를 누름.

2. 표시되는 상단 탭들 중 `Console`을 클릭

3. 다음 코드를 삽입하고 `Enter`를 눌러주면 12시간동안 서버가 끊기지 않고 유지된다.
   - 60초 간격으로 `Colab 연결 끊김 방지`라는 텍스트가 출력되며 연결을 유지함.

```javascript
function ClickConnect(){
  console.log("Colab 연결 끊김 방지"); 
  document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
  console.log("Connnect Clicked - End"); 
};
setInterval(ClickConnect, 60000)
```

