# Model Baseline Insult

Bu proje, Teknofest 2023 Türkçe Doğal Dil İşleme Yarışması kapsamında Türkçe dilindeki metinleri beş kategoriye ayıran bir sınıflandırma problemi için baseline oluşturma çalışmasıdır.

Baseline, bir proje veya problem için başlangıç noktası olarak kullanılan en basit modeldir. Genellikle bir veri setindeki verilerin dağılımını tahmin etmek veya bir algoritmanın başarısını değerlendirmek için kullanılır.

Bu çalışmada, [dbmdz/bert-base-turkish-128k-uncased](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased) modeli kullanılarak daha iyi bir başlangıç noktası elde edilmiştir. BERT tabanlı Türkçe model, Türkçe dilinde önceden eğitilmiş bir versiyonudur. Bu model, text classification, sentiment analysis, NER ve diğer NLP çalışmalarında kullanılabilir. BERT modeli, büyük miktarda metin verisine dayalı olarak eğitilmiş bir makine öğrenimi modelidir ve doğal dil işleme alanında yaygın olarak kullanılmaktadır. Bu önceden eğitilmiş BERT modeli, PyTorch ve Transformers kütüphaneleri ile uyumlu olarak çalışır. Bu sayede Türkçe dilindeki metin sınıflandırma projelerinde kullanılabilecek harika bir imkan sunar.

### Kategori Bazlı Sınıflandırma Performansı

Model, beş farklı kategorideki Türkçe metinleri sınıflandırmak için eğitilmiştir. Bu kategoriler INSULT, OTHER, PROFANITY, RACIST ve SEXIST'tir. Test veri setinde, her bir kategori için precision, recall ve F1-score gibi ölçütler hesaplanarak, modelin sınıflandırma performansı ayrıntılı olarak incelenmiştir. Sonuçlar aşağıdaki tabloda verilmiştir: 


<table><thead><tr><th></th><th>INSULT</th><th>OTHER</th><th>PROFANITY</th><th>RACIST</th><th>SEXIST</th><th>accuracy</th><th>macro avg</th><th>weighted avg</th></tr></thead><tbody><tr><td>Precision</td><td>0.905930</td><td>0.975255</td><td>0.969499</td><td>0.913242</td><td>0.957447</td><td>0.946715</td><td>0.944275</td><td>0.947758</td></tr><tr><td>Recall</td><td>0.924843</td><td>0.941011</td><td>0.930962</td><td>0.985222</td><td>0.961995</td><td>0.946715</td><td>0.948807</td><td>0.946715</td></tr><tr><td>F1-score</td><td>0.915289</td><td>0.957827</td><td>0.949840</td><td>0.947867</td><td>0.959716</td><td>0.946715</td><td>0.946108</td><td>0.946833</td></tr><tr><td>Support</td><td>479</td><td>712</td><td>478</td><td>406</td><td>421</td><td>0.946715</td><td>2496</td><td>2496</td></tr></tbody></table>

Modelin sınıflandırma performansı, kategoriler arasındaki farkları ve modelin hangi kategorilerde daha başarılı olduğunu gösterir. Bu bilgiler, modelin sınıflandırma performansını iyileştirmek için alınabilecek önlemleri veya daha iyi sonuçlar elde etmek için yapılacak değişiklikleri belirlemek açısından önemlidir.


## Confusion Matrix

Confusion matrix, bir sınıflandırma modelinin tahminlerinin gerçek değerlerle karşılaştırılması sonucu elde edilen bir matristir. Bu matris, sınıflandırma modelinin performansını ölçmek için kullanılır.

Bu örnek confusion matrix, beş farklı sınıf için oluşturulan bir modelin performansını göstermektedir. Matrisin satırları gerçek sınıfları, sütunları ise tahmin edilen sınıfları temsil eder. Bu matrisin diyagonalindeki elemanlar doğru tahminleri, diğer elemanlar ise yanlış tahminleri temsil eder.

Sonuç olarak, base modelimizin performansı oldukça iyi görünüyor. Precision, Recall ve F-Score gibi metrikler ile yüksek sonuçlar elde edilmiştir. Ayrıca, confusion matrix'te de görülebileceği gibi, modelimiz çoğunlukla doğru tahminler yapmıştır. Ancak, özellikle RACIST ve PROFANITY sınıflarındaki yanlış sınıflandırmalar daha dikkat çekici. Bu sınıfların daha dengeli bir şekilde öğrenilmesi için modelin daha fazla veriye ihtiyacı olabilir. 


|   |
| - |
|![cf](https://user-images.githubusercontent.com/83168207/229031158-ac952dfc-af40-444f-b51b-cff38fc19b45.png)|


### Genel Bakış (Tests/observation.ipynb)

Bu notebook'ta, veri seti üzerinde yapılan bazı veri ön işleme adımlarına ve veri keşfi analizine yer verilmiştir. Tamamiyle deneysel çalışma amacıyla kullanılan bir notebook'tur. Aşağıdaki adımlar gerçekleştirilmiştir:

- Verileri test amacıyla veri ön işleme için heroku'da deploy ettiğimiz API'ye gönderilerek, metinler veri önişleme işlemine tabi tutulmuştur.
- Veri setindeki duplicated olan gözlem birimleri kaldırılmıştır.
- Veri setindeki kısa metinli gözlem birimleri kaldırılmıştır. (ör: "asda" -> drop)
- Veri setindeki bazı gözlem birimleri, target sınıfları replace edilerek değiştirilmiştir. Bu işlem, "replace_is_offensive" fonksiyonu kullanılarak gerçekleştirilmiştir. Bu fonksiyon, "OTHER" sınıfına ait olan ve aynı zamanda "is_offensive" özelliği de 1 olan gözlem birimlerinin "is_offensive" değerini 0'a dönüştürmektedir.

#### Aşağıdaki tablo, replace işlemi öncesinde ve sonrasındaki sınıf dağılımını göstermektedir.

<div class="markdown prose w-full break-words dark:prose-invert light"><table><thead><tr><th>Target Sınıfı</th><th>is_offensive = 0</th><th>is_offensive = 1</th></tr></thead><tbody><tr><td>INSULT</td><td>-</td><td>2393</td></tr><tr><td>OTHER</td><td>3511</td><td>56</td></tr><tr><td>PROFANITY</td><td>-</td><td>2372</td></tr><tr><td>RACIST</td><td>1</td><td>2016</td></tr><tr><td>SEXIST</td><td>-</td><td>2079</td></tr></tbody></table><div>

## Data Preprocessing Script 

Bu script, bir pandas DataFrame içindeki metin verileri üzerinde çeşitli veri ön işleme adımlarını uygulamak için tasarlanmıştır. Veri ön işleme adımları arasında sayısal metinleri normalleştirme, noktalama işaretlerini kaldırma, Türkçe karakterleri normalize etme, karakterleri küçük harfe dönüştürme, kısa metinleri kaldırma ve "is_offensive" sütunundaki belirli değerleri değiştirme yer alır.

#### Sınıf Özellikleri

<table><thead><tr><th>Değişken</th><th>Açıklama</th></tr></thead><tbody><tr><td><code>df - pd.DataFrame</code></td><td>Veri ön işleme yapılacak pandas DataFrame.</td></tr><tr><td><code>text_column - str</code></td><td>Veri ön işleme yapılacak metin sütununun adı.</td></tr><tr><td><code>words_sw - dict/json</code></td><td>Kaba/küfürlü kelimelerin key/value formatında tutan bir veri yapısı. Aykırı sıkça kullanılan aykırı kelimelerin kısaltmalarını convert etmek maksadıyla oluşturulmuştur.</td></tr></tbody></table>

#### Sınıf Fonksiyonları/İşlevleri: 

<table><thead><tr><th>Fonksiyon</th><th>Açıklama</th></tr></thead><tbody><tr><td><code>preprocess() -&gt; pd.DataFrame</code></td><td>Tüm veri ön işleme adımlarının uygulandığı pandas DataFrame'ini döndürür.</td></tr><tr><td><code>convert_offensive_contractions() -&gt; None</code></td><td>Belirtilen DataFrame sütunundaki kaba/küfür aykırı kelimeleri değiştirir. (Ör: b*k -&gt; bok , aw -&gt; amına koyiyim vb)</td></tr><tr><td><code>normalize_numeric_text_in_dataframe_column() -&gt; pd.DataFrame</code></td><td>Geliştirmek üzere olduğumuz Türkçe Doğal Dil İşleme Kütüphanesini kullanarak bir pandas DataFrame sütununda bulunan sayısal metinleri normalleştirir. (Ör: 2021 yılında, İstanbul toplam nüfusu 15840900 -&gt; iki bin yirmi bir yılında, İstanbul toplam nüfusu on beş milyon sekiz yüz kırk bin dokuz yüz)</td></tr><tr><td><code>mintlemon_data_preprocessing() -&gt; None</code></td><td>Verilen DataFrame sütunundaki metinleri geliştirmek üzere olduğumuz Türkçe Doğal Dil İşleme Kütüphanesinden Normalizer Modülünde çeşitli adımlara tabi tutarak veri ön işleme yapar.</td></tr><tr><td><code>remove_short_text() -&gt; None</code></td><td>Belirtilen DataFrame'den, belirli bir uzunluğun altındaki metin değerlerine sahip gözlem birimlerini kaldırır.</td></tr><tr><td><code>replace_is_offensive() -&gt; None</code></td><td>Belirtilen koşulları sağlayan 'is_offensive' değerlerini değiştirir. (Bağımlı değişken(target/dependent) incelendiğinde sınıf dağılımında aykırılık rastlanmıştır bu nedenle bu fonksiyon yazılmıştır.)</td></tr></tbody></table>


#### Örnek Kullanım

Bu scripti kullanmak için, pandas DataFrame'inizi ve ver ön işleme işleminin uygulanacağı metin sütununun adını belirtmeniz gerekiyor. Daha sonra sınıfın "preprocess()" yöntemini çağırarak tüm ön işleme adımlarını uygulayabilirsiniz.

```python
import pandas as pd
from data_preprocessing import DataPreprocessor

# Önceden yüklenmiş bir pandas DataFrame'i okuyunuz.
df = pd.read_csv("data/teknofest_train_final.csv", sep="|")

# DataPreprocessor sınıfı örneği oluşturunuz.
preprocessor = DataPreprocessor(df, "text")

# Veri ön işleme fonksiyonlarını/işlevlerini uygulayınız.
df = preprocessor.preprocess()

# Verilerdeki duplicated değerleri kaldırınız.
df.drop_duplicates(subset="text", inplace=True)

# İşlenmiş verileri csv olarak kaydediniz.
df.to_csv("data/teknofest_train_final_preprocessed.csv", index=False)
```


## Not: 

* Veri ön işleme aşamasını ayrıntılı incelemek için lütfen [preprocessing-service](https://github.com/Teknofest-Nane-Limon/preprocessing-service) reposuna bir göz atın. 

---
