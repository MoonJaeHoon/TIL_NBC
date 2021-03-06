# 1. 모델 결과 정리

10-Fold CV 중에 User별 마지막 row들로 검증이 이루어지게 세팅되었습니다.

### 1.1 (0609_1609)

- train과 test 합쳐서 학습

- train => holdout split시에 따로 필터 걸지 않았음
- holdout은 test 필터가 걸려있음

- 민용, 진현이형, 재훈 피쳐들을 사용
- 임의로 cat, conti 구별하여 지정해주었음
- userID, Timestamp가 설명변수로서 추가되었음



> holdout 결과

![image-20210610073137498](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610073137498.png)

> 10-fold CV 결과

![image-20210610073730867](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610073730867.png)

> 직접 10-fold Score 확인해보기 (seed=42)

![image-20210610073921863](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610073921863.png)





### 1.2 (0610_0648)

- train만 학습
- train => holdout split시에 **<u>train_only_LB = test로 필터 걸었음</u>**
- holdout도 test 필터가 걸려있음

- 아라, 민용, 진현이형, 재훈 피쳐들을 사용
- 임의로 cat, conti 구별하여 지정하지 않았음.
- **<u>userID, Timestamp를 설명변수에서 제외함.</u>**

> holdout 결과

![image-20210610073125181](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610073125181.png)

> 10-fold CV 결과

![image-20210610073840447](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610073840447.png)

> 직접 10-fold Score 확인해보기 (seed=42)

![image-20210610074025991](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610074025991.png)



### 1.3 (0610_0756)

- train만 학습
- train => holdout split시에 **<u>train_only_LB = problem으로 필터 걸었음</u>**
- holdout은 test 필터가 걸려있음

- 아라, 민용, 진현이형, 재훈 피쳐들을 사용
- **<u>임의로 cat feature 3개로만 골랐음. (problem_number, test_pre, over_300)</u>**
- userID, Timestamp를 설명변수에서 제외함.



> holdout 결과

![image-20210610080842102](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610080842102.png)

> 10-fold CV 결과

![image-20210610080928092](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610080928092.png)

> 직접 10-fold Score 확인해보기 (seed=42)

![image-20210610080954119](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610080954119.png)



### 1.4 (0610_0825)

- train만 학습
- train => holdout split시에 train_only_LB = problem으로 필터 걸었음
- holdout은 test 필터가 걸려있음

- 아라, 민용, 진현이형, 재훈 피쳐들을 사용
- 임의로 cat feature 3개로만 골랐음. (problem_number, test_pre, over_300)
- userID, Timestamp를 설명변수에서 제외함.
- **<u>problem_time_diff_in_user 변수의  threshold를 300초 -> 3600초(1시간)으로  바꿈</u>**
  - 이 변수를 이렇게 처리하니까 LB결과가 매우 안좋아졌음..



> holdout 결과

![image-20210610083051120](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610083051120.png)

> 10-fold CV 결과

![image-20210610083251896](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610083251896.png)

> 직접 10-fold Score 확인해보기 (seed=42)

![image-20210610083323224](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610083323224.png)



### 1.5 (0610_0841)

- train만 학습
- train => holdout split시에 train_only_LB = problem으로 필터 걸었음
- holdout은 test 필터가 걸려있음

- 아라, 민용, 진현이형, 재훈 피쳐들을 사용
- 임의로 cat feature 3개로만 골랐음. (problem_number, test_pre, over_300)
- userID, Timestamp를 설명변수에서 제외함.
- <u>**train 에서 user별 마지막 9 rows만 추출하여 학습함.**</u>



> holdout 결과

![image-20210610084502933](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610084502933.png)

> 10-fold CV 결과

![image-20210610084624672](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610084624672.png)

> 직접 10-fold Score 확인해보기 (seed=42)

![image-20210610084638599](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610084638599.png)



### 1.6 (0610_0848)

- train만 학습
- train => holdout split시에 **<u>train_only_LB 필터 아무것도 안 걺</u>**
- holdout은 test 필터가 걸려있음

- 아라, 민용, 진현이형, 재훈 피쳐들을 사용
- 임의로 cat feature 3개로만 골랐음. (problem_number, test_pre, over_300)
- userID, Timestamp를 설명변수에서 제외함.
- <u>**train 에서 user별 마지막 9 rows만 추출하여 학습함.**</u>



> holdout 결과

![image-20210610085503175](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610085503175.png)

> 10-fold CV 결과

![image-20210610085529821](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610085529821.png)

> 직접 10-fold Score 확인해보기 (seed=42)

![image-20210610085552492](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210610085552492.png)



## 1.7

- train_only 필터 problem으로 해주었음
- 이전 (1.6)에서보다 피쳐가 3개 추가되었음
- 추가된 피쳐는 **`user가 어느 한 test 내부에서 이전 문제에서 다음 문제를 풀기까지의 시간간격`**을 나타내는 변수임
  - `time_elapsed_per_one_problem`, 
  - `time_elapsed_over_300`,
  -  `time_elapsed_to_bins`
  - 가 추가되었음



![image-20210611084922325](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210611084922325.png)

![image-20210611084956825](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210611084956825.png)

![image-20210611085013923](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210611085013923.png)

## 1.8

- 1.7에서 SMOTE를 적용하였음 (다른건 그대로)



![image-20210611090306835](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210611090306835.png)

![image-20210611090332671](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210611090332671.png)

![image-20210611090352089](0610%20%EC%A0%95%EB%A6%AC.assets/image-20210611090352089.png)



### LB SCORE

|                                                    | LB - Accuracy |     LB - AUC      |  holdout   | 10-fold (CV)<br />(Stratified X) | Finalize이후<br/>10-fold |
| :------------------------------------------------: | :-----------: | :---------------: | :--------: | :------------------------------: | :----------------------: |
|                1.1<br />(0609_1609)                |    0.7419     |      0.7947       |   0.8123   |              0.8090              |          0.8144          |
|                1.2<br />(0610_0648)                |    0.7258     |      0.7991       |   0.8264   |              0.8129              |          0.8264          |
|                1.3<br />(0610_0756)                |    0.7366     |      0.8046       |   0.8431   |            **0.8167**            |          0.8381          |
|                1.4<br />(0610_0825)                |    0.7392     |      0.7982       | **0.8511** |              0.8160              |        **0.8424**        |
|                1.5<br />(0610_0841)                |    0.6935     |      0.8014       |   0.8412   |              0.8074              |          0.8329          |
|                1.6<br />(0610_0848)                |    0.7473     |      0.8011       |   0.8334   |              0.8159              |          0.8382          |
|                        1.7                         |    0.7446     |      0.7999       |   0.8138   |              0.8233              |          0.8168          |
|                        1.8                         |    0.7554     |      0.7988       |   0.8129   |              0.8232              |          0.8160          |
|     boosting = `gbdt`, <br />with interaction      |               |      0.8016       |   0.8182   |              0.8072              |         0.81414          |
|      boosting = `gbdt`, <br />not interaction      |               | **<u>0.8072</u>** |   0.8174   |              0.8062              |         0.81314          |
|     boosting = `dart`, <br />with interaction      |               |         ?         |   0.8214   |              0.8065              |         0.81502          |
|      boosting = `dart`, <br />not interaction      |               |      0.8007       |   0.8210   |              0.8072              |         0.81453          |
| boosting=`gbdt`<br />DL 7 Feature with interaction |               |                   |   0.8192   |              0.7919              |         0.82012          |



> **경향 정리**
>
> - userID와 Timestamp는 변수에서 제거하는 것이 괜찮아 보인다.
> - 전반적으로 train data 구축시 미리 filter를 걸어서 일부 데이터를 추출하고 학습하는 것이 좋았음
>   - 현재까진 problem filter를 거는 것이 가장 좋았음
>
> - time_diff 변수는 300을 기준으로 한 것이 좋아보임. (다른 thres도 실험해볼 필요는 있음)
> - 데이터를 적게 사용할수록 정확도가 희생되고 AUC는 오르는 경향이 있음
> - LB를 대표하는 데이터셋으로서 아직 괜찮은 어떤 데이터셋을 구축하지 못했다.
> - SMOTE를 사용하면 정확도는 보다 상승하지만, AUC가 희생된다.
> - 변수 개수가 늘어날수록? 혹은 특정 변수가 추가될 때, Generalization 측면에서 이점이 거의 없어진다.

