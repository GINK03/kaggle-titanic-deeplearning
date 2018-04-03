# kaggle titanic deeplearning

## データセットの用意
kaggle apiでダウンロードします  

```console
$ kaggle competitions download -c titanic
```

$HOME/.kaggle/competitions/titanicにダウンロードされます

## 方針
いろんなフィールドがありますが、シンプルな３つの全結合層で表現しようとすると、入力に名前の文字列などは扱いづらく、そのため、代表的な扱いやすいパラメータに限定して取り扱います  

使用する特徴量は、"Ticket", "Name", "Cabin"以外のものすべてです（後述しますが、うまくNameの特徴量を使うことで精度はもっとあがります）  

数値として扱うものは、"PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"で、他をカテゴリ変数とします。  

数値がないときは、数値であっても"カテゴリ変数で何もない"を表現して、あらたに特徴量に加えます。  

この作業をするのが下記のプログラムになります  
```console
$ python3 10-feat-prepare.py --step1
$ python3 10-feat-prepare.py --step2
```

## モデル
全結合相を3つ組み合わせたモデルで、中間の活性化関数をreluで、最後はsigmoidで生き残ったかどうかを予想します。　　　

```python
x = input_tensor
x = Dense(200, activation='relu')(x)
x = Dense(200, activation='relu')(x)
x = Dense(200, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)
```

## 学習
以下のコマンドで学習できます  
```console
$ python3 20-train.py --train
```
trainをさらに分割してvalidationに用いてますが、kaggleはtestに正解が付かないので、しょうがないです  

validationの精度を見ていくと、最良値で、82%程度で、これは、別の[KaggleのTitanicのTFでの実装](https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic)に迫るものです。  

基本的な方針としてはこれで良さそうです。  

## xgboostにはかなわない
92%程度の正解率です。xgboostは。

https://www.kaggle.com/tanlikesmath/titanic-xgboost　　

ゴリゴリにチューンすればなにかDeepでもあるかもですが、そこまではやらないこととします。  

