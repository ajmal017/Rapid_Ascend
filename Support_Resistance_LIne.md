## SupportLine & ResistanceLine

### **<span style= "color:red">  = Oversold, Overbought </span>**

1. <u>내가 정한 매수가(선)</u>이 **지지선**의 **최저가**인가
2. <u>내가 정한 매도가(선)</u>이 **저항선**의 **최고가 / 고가** 인가

| Finding Signal | Setting Method                                               | Remark                                   |
| -------------- | ------------------------------------------------------------ | ---------------------------------------- |
| Support        | 인공지능 / **RSI** /                                         | **다양성은 없어도 정확도가 높아야한다.** |
| Resistance     | **지정 수익제** + 비지정 수익제 / 인공지능 + 퀀트 / 인공지능은 다양성을 보장하지 못한다.  / 지정 수익제로 n% 수익을 꾀하고 | **다양성과 정확도가 높아야한다.**        |

| Model_Num | Setting                                      | Result                             |
| --------- | -------------------------------------------- | ---------------------------------- |
| _124      | **crop_size_scale 데이터셋 사용** 작은 idl값 |                                    |
| _125      | 변곡점 다음을 라벨링한다.                    | crop_size_scale 정확도가 떨어진다. |
| _126      | 60 + none max pooling                        |                                    |
| _127      | _125 max pooling                             |                                    |

* 지지선을 다 찾을 필요가 없다. 가장 정확한 지지선만 찾는다.

  

## Humachine

손절가 = 매수가의 1% 이하로 떨어지면 / 들어가는 시점에서 이전 저점의 밑으로 떨어지면

매도가 = **이전 고점 언저리**

인공지능을 이용해 지지 저항을 찾을 수 없다면 **Human + Machine**으로 가자



* **PyQT UI를 사용한 자동거래 ㄱㄱ, 거래 완료될 때까지 못기다리겄다..** 
  * 항상 7개의 후보 코인을 유지 (7개의 탭)
  * 매수 등록 취소는 가능한데 / **<u>매도 등록 취소하고 재등록</u>**이 안돼네?



## Logic

전 아랫 봉우리 > 전전 아랫 봉우리 < 전전전 아랫 봉우리 **(<span style="color:blue">V shape</span>)**

`+` Stochastic Setting을 변경해서 지지 고저에 가장 최적화시킨다.

**최저점을 찾는 setting**



## Reference

웹크롤링 - iframe 처리하기 : https://dejavuqa.tistory.com/198

Stochastic RSI : https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi



