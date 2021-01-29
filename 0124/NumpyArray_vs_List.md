## numpy array와 list의 메모리 효율 차이



1. `list`는 가져올때 한번 더 추가적인 과정을 거침
2. `array`의 원소들은 메모리의 크기가 일정하기 때문에, 데이터를 저장하는 공간(메모리)를 사용하는 데 이점이 있다.



1)

-5부터 256까지의 값이 메모리의 `static`한 공간에 있음,

리스트는 값들을 저장하면 그 값들의 주소값을 저장해서 갖고있게 됨.

따라서 `array`는 한단계만 거치면 되는 것을

`list`는 인덱싱으로 접근 -> 그 값을 저장해놓은 주소값을 통해 값을 가져오기 2단계를 거쳐야한다.





2) `array`의 원소들은 다음과 같이 각각 메모리의 크기가 일정하기 때문에 데이터를 저장하는 공간 활용에 이점이 있다.

<img src="NumpyArray_vs_List.assets/image-20210130060813184.png" alt="image-20210130060813184" style="zoom:80%;" />






> **결론**
>
> - list보다 array의 메모리 효율이 좋다.
>
>   - `numpy`는 일반적인 `array`처럼 연속된 메모리 공간으로 할당하고,
>
>   - 파이썬 `list`는 각 위치에 실제 `value`값으로 연결되는 메모리 주소를 저장하고 있는 것이기 때문에,
>
>   - 주소값을 담아두는 공간이 추가적으로 필요없기 때문에 `array`가 `list`보다 메모리 효율이 좋다고 할 수 있습니다.
>   - 또한, `array`는 type을 미리 지정해두기 때문에 각 원소의 `type`을 따로 지정하지 않아도 된다는 점을 장점으로 볼 수 있습니다.
>
> 
>
> - `list`의 장점은 유연함.
>   - `list`는 포인터를 저장하고 있기 때문에 memory를 훨씬 많이 사용하는 반면, 서로 다른 타입의 원소를 저장할 수 있고, `list`의 길이를 유연하게 조정할 수 있다는 장점이 있습니다.





## 리스트와 배열 원소 메모리주소 차이



- 리스트를 인덱싱했을 때와 

<img src="https://cphinf.pstatic.net/mooc/20210125_93/1611543130293nDble_PNG/_2021-01-25__11.51.54.png" alt="_2021-01-25__11.51.54.png (912×406)" style="zoom: 50%;" />



- 리스트에서의 결과와 array에서의 결과가 왜 이렇게 다르게 나오는 것일까? 아래 예시를 보도록 하자.



<img src="https://cphinf.pstatic.net/mooc/20210126_93/16116214840599uvUE_PNG/%2C_2021-01-26_09-32-37.png" alt="img" style="zoom:50%;" />

- 분명 -5부터 256 사이의 `python int` 객체들은 각각 같은 메모리 주소를 가진다고 배웠는데 왜 다른 결과가 나오는 걸까요?

  - 이는 `python native integer`가 아니라 `numpy object`이기 때문에 해당되지 않는 것입니다.

  - 따라서, `int type`이 아니라 `numpy.int type`이기 때문에 `id(a[0])`의 주소값은 다음과 같이 호출할 때마다 바뀌게 됩니다.

    ```python
    print(id(a[0]))
    print(id(a[0]))
    ```

    ```
    140452414823632
    140451605401488
    ```

    

- 먼저, Python에서 id는 객체의 메모리 주소값이고, 이는 객체가 존재하는 동안에만 unique하다는 것을 기억해야 합니다.
- 즉, 한 객체가 사라지고 다른 객체가 만들어졌을 때 이 두 객체는 같은 id를 가질 수도 있습니다. (이는 메모리를 재사용하기 때문입니다.)



- `a[0] is b[-1]`에서는 이 한 줄의 코드 내에서 `a[0]`가 아직 사라지지 않은 상태이기 때문에 해당 메모리를 차지하고 있고, `b[-1]`은 다른 메모리 주소를 가지게 되어 False를 반환하게 됩니다.
- `id(a[0]), id(b[-1])`에서는 한 줄의 코드지만 `id(a[0])`가 호출되었다가 사라지고 해당 메모리 주소가 비어있기 때문에 다시 해당 메모리 주소를 `id(b[-1])`가 가질 수 있게 된 것입니다.



> 참조 : https://stackoverflow.com/questions/52096582/how-unique-is-pythons-id#:~:text=The%20id%20is%20unique%20only,implementation%20provided%20by%20python.org



## 유용한 함수들

> np.zeros
np.ones(shape=(m,n), dtype=np.int8)
np.empty(shape=(m,n), dtype=np.int8)
np.zeros_like(array)
np.ones_like(array)
np.empty_like(array)
기존 ndarray의 shape만큼 1,0 또는 empty array를 반환