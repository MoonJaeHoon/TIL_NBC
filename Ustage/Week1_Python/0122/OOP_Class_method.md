# Object Oriented Program
## 1. 객체 (Object)

Python에서 생성되는 모든 것이 **`객체`**입니다.

모든 객체들은 `type`, `attribute`, `method`를 가지고 있습니다.

- 이중에서 type은 말그대로 변수의 형태를 분류해놓은 것이다.
- attribute와 method는 제대로 알아놓으면 쉽지만, 모르는 이들은 헷갈릴 수도 있기에 정리하려 한다.



### 1.1 attribute(속성)

- 속성은 객체의 상태/데이터를 뜻하는 단어입니다.
  - ex) <객체>.<속성>
- 메소드는 객체에 적용하는 행동을 뜻합니다.
  - ex) <객체>.<조작법>()



> `attributes` vs 	`method`

- `complex` 타입의 객체를 예로 들어 살펴보겠다.
  - `객체.imag`				 : `attribute`
  - `객체.conjugate()`  : `method`

```python
complex_number = 10+4j
```

```python
complex_number.imag	# 허수부분 attributes
```

```
4.0
```



```python
complex_number.conjugate() # 켤레복소수 method
```

```
(10-4j)
```



## 2 객체지향프로그래밍 (OOP)

- `Object` (위에서 생성한 complex_number와 같은)가 중심이 되는 프로그래밍이다.

- 프로그램을 유연하고 변경이 용이하게 만들기 때문에 대규모 소프트웨어 개발에 많이 사용된다.
  - 코드를 보다 직관적으로 짤 수 있고,
  - 코드를 재차 활용하거나, 변경할 때 매우 용이하다



> 간단한예시를 보면서 이해하자.

```python
class SoccerPlayrer(object):
    ### 1) attribute ###
    def __init__(self,name,position,back_number):
        self.name  = name
        self.position = position
        self.back_number = back_number
        
    ## 2) method ##
    def talk(self):
        print("안녕")
    def check_name(self, new_name):
        print(f"선수의 이름을 변경합니다 : From {self.name} to {new_name}")
        self.name = new_name
```



```python

IU = SoccerPlayer("IU","MF",10)
## attribute
print(IU.name)

## 메소드
print(IU.talk())
IU.check_name("JH")
```

```
10
안녕
선수의 이름을 변경합니다 : From {IU} to {JH}
```



​	1) `attribute`는 `__init__` 과 `self`를 함께 써서 만들면 됩니다. 

​	2) `method`는 기존 함수를 정의하는 것과 같아 보이지만, 반드시 `self`를 인자로 추가해야만 `class`의 함수로서 인정이 됩니다.





> **여기까지 정리**

- 객체(Object) : 자신 고유의 **속성(attribute)**을 가지며 클래스에서 정의한 **행위(behavior)**를 수행할 수 있다.
- 클래스(Class)  : 공통된 속성(attribute)과 행위(behavior)를 정의한 것으로 객체지향 프로그램의 기본적인 **사용자 정의 데이터형(user-defined data type)**
- 인스턴스(Instance)  : 특정 `class`로부터 생성된 해당 클래스의 실체/예시
- 속성(Attribute)  : 클래스/인스턴스가 가지는 속성(값/데이터)
- 메서드(Method)  : 클래스/인스턴스에 적용 가능한 조작법(method) & 클래스/인스턴스가 할 수 있는 행위



## 3. 메서드의 종류



```python
class MyClass:
    
    # 1) 인스턴스메서드
    def instance_method(self):
        return self
    
    # 2) 클래스메서드
    @classmethod
    def class_method(cls):
        return cls
    
    # 3) 스태틱메서드
    @staticmethod
    def static_method(arg):
        return arg
```



### 3.1 Instance method

- 이제까지 위에서 생성해왔던 **`인스턴스가 사용할 메서드`**가 바로 인스턴스 메서드이다.
- 클래스 내부에 정의되는 메서드의 기본값이 인스턴스 메서드이기 때문에 따로 `decorator`가 필요없다.
- 함수를 정의할 때 `self`를 인자값으로 받기만 하면 인스턴스 메서드로서 인정된다.

### 3.2 Class method

- `@classmethod` 데코레이터를 붙여서 정의하고,
- 클래스 메서드는 첫번째 인자로 `cls`(파이썬의 클래스)를 받게 하면 된다.

### 3.3 Static method

- `@staticmethod` 데코레이터를 붙여서 정의하고,
- 스태틱 메서드는 인자로 `self`, `cls` 무엇도 전달되지 않는다.



>  다음 예시들을 보자.

```python
# 1) 인스턴스 메서드
mc = MyClass()
mc.instance_method()
```

```
<__main__.MyClass at 0x1520f4434c0>
```



```python
# 2) 클래스 메서드
## 인스턴스에서 클래스 메서드에 접근 가능하지만, 권장되지 않는다.
mc.class_method()
```



```python
# 3) 스태틱 메서드 # 에러발생
mc.static_method()
```

```
TypeError                                 Traceback (most recent call last)
<ipython-input-40-c9cd4dcf9ba4> in <module>
      1 # Error => 첫 번째 인자가 없다. 위와 같이 자동으로 첫 번째 인자로 들어가는 것이 없습니다.
----> 2 mc.static_method()

TypeError: static_method() missing 1 required positional argument: 'arg'

```



- 위 `class Myclass`에서 스태틱 메서드를 정의함에 있어 인자값을 하나 받기로 지정했는데, 메서드를 수행할 때 인자값을 받은 것이 없기 때문에 인자를 넣지 않았다는 오류가 발생한 것입니다.
- 아래와 같이 인자값을 넣으면 정상적으로 수행됨을 볼 수 있다.

```python
# 스태틱 메서드 # 에러발생
mc.static_method(1)
```

```
1
```





## 4. 상속

- 부모 클래스의 모든 속성이 자식 클래스에게 상속이 가능하여 코드 재사용성을 매우 높일 수 있습니다.

- 이는 클래스의 가장 큰 특징이자 장점입니다.



### 4.1 super()

> 다음 예시를 봅시다.

```python
class Parent:
    population = 0
    
    def __init__(self, name='사람'):
        self.name = name
        Person.population += 1
        
    def talk(self):
        print(f'반갑습니다. {self.name}입니다.')
        
    def check(self, new_name):
        print(f'{self.name}와 {new_name}은 똑같다 : {self.name==new_name}')
```

- 이렇게 먼저 생성한 클래스를 상속받아 `Student` 클래스를 만들어 보겠습니다.
- 이를 상속받아 만들어진 `Student` Class가 `자식클래스`, `Parent` Class가 `부모 클래스`입니다.

```python
class Student(Parent):
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
    
    # 상속에선 super()를 많이 씁니다,
    # 부모클래스의 메서드를 상속받아 그대로 쓰면서,
    # 코드를 추가하려고 하는데, 똑같을 부분을 일일이 코드치기 힘드니까요
    def check(self, new_name,new_name2):
        super().check(new_name2)	# 여기서 부모클래스의 check는 다 받아옴.
        # 해당 메서드에 추가할 코드
        print(f"추가적으로 검사합니다, {self.name}과 {new_name2}은 똑같다 : {self.name==new_name2}")
        print('='*30)
```

```python
# 학생을 만들어봅시다.
s1 = Student('김부비', '20210302')
print(s1.name)
print(s1.student_id)
```

```
김부비
20210127
```



```python
### 자식클래스에서 따로 정의하지 않았지만,
## 부모 클래스에서 정의된 메서드를 호출 할 수가 있습니다.
s1.talk()
```

```
반갑습니다. 박학생입니다.
```



> **super**()

```python
p1 = Parent('김자비')
s1 = Student('김부비', '20210302')
```

```python
p1.check('김자비')
p1.check('타일러')

```

```
김자비와 김자비은 똑같다 : True
김자비와 타일러은 똑같다 : False
```



```python
s1.check('김자비','타일러')
s1.check('김부비','바비')
```

```
김부비와 타일러은 똑같다 : False
추가적으로 검사합니다, 김부비과 타일러은 똑같다 : False
==============================
김부비와 바비은 똑같다 : False
추가적으로 검사합니다, 김부비과 바비은 똑같다 : False
==============================
```





> 추가적으로, 함께 알아두면 좋을 코드1 (상속관계 검사)

```python
print(issubclass(Student, Person))
print(isinstance(s1, Student)) # 인스턴스인지 확인하는 코드
print(isinstance(s1, Person))

```

```
True
True
True
```



> 함께 알아두면 좋을 코드2 (타입 검사)

```python
print(type(s1) is Student)
print(type(s1) is Person) # 상속관계인지 확인하는 게 아니라 타입을 확인하는 것이기 때문에 False
```

````
True
False
````



### 4.2 Overriding (메서드 오버라이딩)

- 부모클래스를 자식클래스가 상속받을 때, 부모클래스의 메서드를 재정의하는 것입니다.

> 바로 예시를 보겠습니다.

```python
class Parent:
    population = 0
    
    def __init__(self, name='사람'):
        self.name = name
        Person.population += 1
        
    def talk(self):
        print(f'반갑습니다. {self.name}입니다.')
        
    def check(self, new_name):
        print(f'{self.name}와 {new_name}은 똑같다 : {self.name==new_name}')
```



```python
class Student(Parent):
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
    
    def check(self):
        print(f"메서드 오버라이딩으로 새로 정의했습니다. 이전에 당신이 알던 부모클래스의 check메서드는 사라졌습니다.")

```
```
메서드 오버라이딩으로 새로 정의했습니다. 이전에 당신이 알던 부모클래스의 check메서드는 사라졌습니다.
```



### 4.3 다중상속

- 두개 이상의 클래스를 상속받는 경우에 다중상속이라고 합니다.

```python
class Person:
    
    def __init__(self, name):
        self.name = name
        
    def talk(self):
        print('사람입니다.')
        
# Person을 상속받은 Mom class #        
class Mom(Person):
    gene = 'XX'
    
    def swim(self):
        print('첨벙첨벙')
        
# Person을 상속받은 Dad class #        
class Dad(Person):
    gene = 'XY'
    
    def walk(self):
        print('씩씩하게 걷기')
```

```python
mommy = Mom('박엄마')
mommy.swim() # Moomy는 Dad 클래스처럼 walk 메서드는 없습니다.
print(mommy.gene)
```

```
첨벙첨벙
XX
```



```python
daddy = Dad('김아빠')
daddy.walk() # daddy는 Mom클래스처럼 swim 메서드는 없습니다.
print(daddy.gene)
```

```
씩씩하게 걷기
XY
```



> 다중상속의 예시

```python
## 다중상속 
# Mom class와 Dad class 다중상속받은 FirstChild 클래스 정의
class FirstChild(Mom, Dad):
    
    def cry(self):	# 메서드 오버라이딩
        print('응애')
        
    def walk(self):	# 메서드 오버라이딩
        print('아장아장')
```

```python
baby = FirstChild('이아가')
baby.cry()
baby.swim()
baby.walk()
```

```
응애
첨벙첨벙
아장아장
```



```python
baby.gene
```

```
'XX'
```

- 이런 결과가 나오는 이유는 상속 순서에 있다.
- `baby = FirstChild(Mom,Dad)`가 아니라 `baby = FirstChild(Dad,Mom)`와 같이 정의했다면
- `baby.gene`의 결과값은 ``'XY'``를 반환한다.





### 4.4 다형성

> 추가적으로, 알고 있어야 할 **`다형성`**에 대해 간단히 짚고 넘어가자.

```python
class Person:
    
    def __init__(self, name):
        self.name = name
        
    def talk(self):
        print('아직 정의되지 않았습니다.')
        
# Person을 상속받은 Mom class #        
class Mom(Person):
    gene = 'XX'
    
    def talk(self):
        print('나는 엄마야')
        
# Person을 상속받은 Dad class #        
class Dad(Person):
    gene = 'XY'
    
    def talk(self):
        print('나는 아빠야')
```

- `다형성`이란 위와 같이 같은 클래스를 상속받아 만들어진 자식클래스가 서로 같은 메소드명 이지만,
- 다르게 동작할 수 있게 로직을 새로 작성할 수 있는 성질을 말합니다.

```python
m = Mom()
d = Dad()
```

```python
m.talk()
d.talk()
```

```
나는 엄마야
나는 아빠야
```



## 4.5 Visibility (가시성)

#### # Visibility Overview

> ###### 객체의 정보를 볼 수 있는 레벨(User Levele)을 조절
>
> ###### 누구나 객체 안의 모든 변수를 볼 필요는 없지 않나? 라는 의문에서
>
> ​	1) 객체를 사용하는 사용자가 임의로 정보 수정
>
> ​	2) 필요없는 정보에는 접근할 필요가 없음
>
> ​	3) 만약 제품으로 출시한다면? 소스의 보호



**<u>Encapsulation</u>**

- 캡슐화 또는 정보 은닉 (Information Hiding)
- Class를 설계할 때, 클래스 간 간섭/정보공유의 최소화
- 캡슐을 던지듯, 인터페이스만 알아서 써야한다고 이해하면 좋다.



> 첫번째 예시를 보겠습니다.

```python
# Visibility Example 1
# Product 객체를 Inventory 객체에 추가
# - Inventory에는 오직 Product 객체만 들어감
# - Inventory에 Product가 몇 개인지 확인이 필요
# - Inventory에 Product items는 직접 접근이 불가
```

```python
class Product(object):
    pass
 
class Inventory(object):
    def __init__(self):
        self.__items = [] ## Private 변수로 선언, 타 객체가 접근을 못함	## Encapsulation
        
    def add_new_item(self, product):
        if type(product) == Product:
            self.__items.append(product)
            print("new item added")

        else:
            raise ValueError("Invalid Item")

    def get_number_of_items(self):
        return len(self.__items)
```



Input :

```python
my_inventory = Inventory()
my_inventory.add_new_item(Product())
my_inventory.add_new_item(Product())
```

Output :

```
new item added
new item added
```



```python
print(my_inventory.get_number_of_items())
```

```
2
```



```python
print(my_inventory.__items) #접근불가
print(my_inventory.items) #접근불가
```

```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-46-d9e074b71d37> in <module>
----> 1 print(my_inventory.__items) #접근불가
      2 print(my_inventory.items) #접근불가

AttributeError: 'Inventory' object has no attribute '__items'

```

- Encapsulation에 의해 attribute에 접근이 불가능한 것을 볼 수 있습니다.

- 다음으론 private 에 접근을 허용하게 하는 예제를 보자.



> 2번째 예제를 보겠습니다.

```python
# Visibility Example 2
# Product 객체를 Inventory 객체에 추가
# Inventory에는 오직 Product 객체만 들어감
# Inventory에 Product가 몇 개인지 확인이 필요
# Inventory에 Product items 접근 허용
```

```python
class Product(object):
    pass

class Inventory(object):
    def __init__(self):
        self.__items = []

    @property
    def items(self):
        return self.__items
    
    def get_number_of_items(self):
        return len(self.__items)
```

```python
my_inventory = Inventory()
items = my_inventory.items
```

- `property decorator`가 붙은 함수를 추가해주니, 이 덕분에 마치 함수를 변수처럼 호출하는 것이 가능해졌습니다.



```python
items.append(Product())
items.append(Product())
print(my_inventory.get_number_of_items())
```

```
2
```



- 그런데 한가지 이상한 점이 있다.

- `__items`과 같이 더블언더바를 통해 private 기능을 가능케 하는 것인데, 이와 같이 private에 접근이 가능해지면 사실상 private의 기능이 없는 것 아닌가?

- 이러한 파이썬의 Security 문제에 대한 논의가 다음에 있다.

  > 참조1 : https://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes

  - 내용을 간단하게 이해해보자면, 결국 파이썬은 자바처럼 `private`으로 `instance`에 대한 접근을 막지 못하는 것이 사실이다.
  - 하지만 더블언더바는 User Level에서 인스턴스의 `private/public` 구별을 위한 용도, 혹은 해당 인스턴스가 이 클래스 안에서만 쓰이는 것이다! 라고 알려주는 `indicator` 역할은 하는데 의미가 있다고 한다.

  



## 5. 결론



### 5.1 메서드

#### 5.1.1 인스턴스

- 인스턴스는 위에서 언급한 3가지 메서드 모두에 접근할 수 있다.
- 하지만, 인스턴스가 할 행동은 `인스턴스 메서드`로서 정의해 놓고, 다른 `클래스 메서드`와 `스태틱 메서드`는 **호출하지 않는 것**이 약속이다.

#### 5.1.2 클래스

- 클래스 또한 3가지 메서드에 모두 접근할 수 있습니다.
- 하지만 클래스에서 `인스턴스 메서드`는 **`호출하지 않는 것`**이 약속이다.
- 클래스 메서드와 정적 메서드는 다음과 같은 규칙에 따라 설계해놓는다고 생각하면 된다.
  - 클래스 자체(cls)와 그 속성에 접근할 필요가 있다면 클래스 메서드로 정의한다.
  - 클래스와 그 속성에 접근할 필요가 없다면 정적 메서드로 정의한다.



### 5.2 상속

- `Overriding` (메서드 오버라이딩)을 통해 부모클래스의 메소드를 자식클래스가 상속받을 때 수정할 수 있다는 것은 이후 머신러닝과 딥러닝 모델 클래스를 상속받아 입맛대로 바꿀 수 있다는 것에 이점이 있을 것이다.
- 다중상속은 두 개 이상의 부모클래스로부터 상속받아 자식클래스를 생성할 수 있다는 것이며, 만약 부모클래스 2개가 같은 `attribute`를 가진다면 먼저 인자로 받은 클래스의 `attribute`를 따라간다는 것을 잊지 말자.
- 다형성은 같은 부모클래스로부터 상속받은 자식클래스들이 같은 메소드명을  수행함에 있어서 다른 결과를 낼 수 있게 로직을 바꿀 수가 있는 성질을 일컫는 말입니다.



### 5.3 가시성

- `Security`를 고려하여 `Encapsulation`을 이용하는 것인데 `private`으로 지정하기 위해 더블언더바 `__attribute`를 사용한다.
- `property decorator`를 함수와 함께 이용하여 `private`이더라도 접근하게 만들 수가 있다.