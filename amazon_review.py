

#    RATING PRODUCT  & SORTING REVIEWS IN AMAZON

"""
        İŞ PROBLEMİ

    E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde
    hesaplanmasıdır.Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak,
    satıcılar için ürünün öne çıkması ve satın alanlar için sorunsuz bir alışveriş deneyimi demektir.
    Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması olarak karşımıza çıkmaktadır.
    Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
    hemde müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar
    satışlarını arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.


        VERİ SETİ HİKAYESİ

    Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
    Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.


    reviewerID     :Kullanıcı ID’si
    asin           :Ürün ID’si
    reviewerName   :Kullanıcı Adı
    helpful        :Faydalı değerlendirme derecesi
    reviewText     :Değerlendirme
    overall        :Ürün rating’i
    summary        :Değerlendirme özeti
    unixReviewTime :Değerlendirme zamanı
    reviewTime     :Değerlendirme zamanı Raw
    day_diff       :Değerlendirmeden itibaren geçen gün sayısı
    helpful_yes    :Değerlendirmenin faydalı bulunma sayısı
    total_vote     :Değerlendirmeye verilen oy sayısı


"""

#       PROJE GOREVLERİ

#############################################################
#GOREV 1: AverageRating’i güncel yorumlara göre hesaplayınız ve
# var olan average rating ile kıyaslayınız
##############################################################

#Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
#Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
#İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head(10)
df.shape
df.columns
df.isnull().sum()
df.dtypes
df.info()
df.describe().T

# Adım 1: Ürünün ortalama puanını hesaplayınız.

df["overall"].mean()

# Adım 2:  Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.

# •reviewTime değişkenini tarih değişkeni olarak tanıtmanız

df["reviewTime"].dtypes
df["reviewTime"] = df["reviewTime"].apply(pd.to_datetime)
df.dtypes

# •reviewTime'ın max değerini current_date olarak kabul etmeniz

from datetime import datetime
import datetime as dt
df["reviewTime"].max()
current_date = df["reviewTime"].max()

#ya da
current_date = dt.datetime(2014, 12, 7) #reviewTime'ın max değerini
type(current_date)

# •her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek
# yeni değişken oluşturmanız ve gün cinsinden ifade edilen değişkeni quantile fonksiyonu ile
# 4'e bölüp (3 çeyrek verilirse 4 parça çıkar)çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
# Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.


df["days_pass"] = (current_date - df["reviewTime"]).dt.days  #dt.days yazmazsak cıktının sonunda 137 days yazıyor ve tipini timedelta64[ns] gösteriyor
df.head(10)
df.dtypes

df["days_pass"].quantile([0.25, 0.50, 0.75])
#q1<280
#q2 280-430 arası
#q3 430-600 arası
#q4>600
df.loc[df["days_pass"] <= 280, "overall"].mean() * 28/100 + \
df.loc[(df["days_pass"] > 280) & (df["days_pass"] <= 430), "overall"].mean() * 26/100 + \
df.loc[(df["days_pass"] > 430) & (df["days_pass"] <= 600), "overall"].mean() * 24/100 + \
df.loc[(df["days_pass"] > 600), "overall"].mean() * 22/100


# Adım 3:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
df.loc[df["days_pass"] <= 280, "overall"].mean() * 28/100
df.loc[(df["days_pass"] > 280) & (df["days_pass"] <= 430), "overall"].mean() * 26/100
df.loc[(df["days_pass"] > 430) & (df["days_pass"] <= 600), "overall"].mean() * 24/100
df.loc[(df["days_pass"] > 600), "overall"].mean() * 22/100

"""
son 280 günde verilen puanların ortalaması digerlerinden daha fazla.Gecen gün sayısı arttıkca verilen puanlamanın ortalaması düsmektedir,
yani müsteriler üründen daha memnun kalmıs ve daha yüksek puanlar vermiştir
"""
#############################################################
#GOREV 2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
##############################################################

# Adım 1:  helpful_no değişkenini üretiniz.

#•total_vote bir yoruma verilen toplam up-down sayısıdır.
#•up, helpful demektir.
#•Veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
#•Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.columns

# Adım 2:  score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz

#•score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için
# score_pos_neg_diff, score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.

def score_pos_neg_diff(yes, no):
    return yes - no

def score_average_rating(yes, no):
    if yes + no == 0:
        return 0
    return yes / (yes + no)

def wilson_lower_bound(yes, no, confidence=0.95):
    n = yes + no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

#•score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.

df["score_pos_neg_diff"]= df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                x["helpful_no"]), axis=1)

#•score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)


#•wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)

df[["overall", "total_vote","helpful_yes", "helpful_no", "score_pos_neg_diff",
    "score_average_rating", "wilson_lower_bound"]].head(20)

df.head(20)


# Adım 3:  20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.

df["score_average_rating"].sort_values( ascending=False).head(20)
df[["score_average_rating", "helpful_yes", "helpful_no", "wilson_lower_bound"]] \

.sort_values("score_average_rating", ascending=False).head(20)

"""
score_average_rating'e göre ilk 20 yorumu sıraladıgımızda evet hepsinin puanı 1 fakat 
burada frekans bilgisini gözardı ederek sıralama yapmıs oluyoruz yani 
kullanıcıların daha faydalı buldugu yorumları baskılamıs,kacırmıs oluyoruz
"""
#•wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.

df["wilson_lower_bound"].sort_values( ascending=False).head(20)

df[["helpful_yes", "helpful_no", "score_average_rating", "wilson_lower_bound"]] \

.sort_values("wilson_lower_bound", ascending=False).head(20)

#•Sonuçları yorumlayınız

"""
wilson_lower_bound yöntemiyle hem oran hem de frekans bilgisini eszamanlı gözönünde bulundurarak skor sıralaması yapmıs olduk.

 7           0               1.00000             0.64567
14           2               0.87500             0.63977  
bu satırlar gözüme takıldı,ikinci satırdaki yorum daha fazla kişi tarafından 
faydalı bulunmasına ragmen digeri onun önüne gecmiş.

"""



