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

<table><thead><tr><th>Target Sınıfı</th><th>is_offensive = 0 (önce)</th><th>is_offensive = 1 (önce)</th><th>is_offensive = 0 (sonra)</th><th>is_offensive = 1 (sonra)</th></tr></thead><tbody><tr><td>INSULT</td><td>-</td><td>2393</td><td>-</td><td>2393</td></tr><tr><td>OTHER</td><td>3511</td><td>56</td><td>3567</td><td>-</td></tr><tr><td>PROFANITY</td><td>-</td><td>2372</td><td>-</td><td>2372</td></tr><tr><td>RACIST</td><td>1</td><td>2016</td><td>-</td><td>2017</td></tr><tr><td>SEXIST</td><td>-</td><td>2079</td><td>-</td><td>2079</td></tr></tbody></table>


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

# Hyperparametre Tuning Yöntemleri ve Uygulamaları

Bu GitHub reposu, 2023 Teknofest Yarışması kapsamında gerçekleştirilen Türkçe doğal dil işleme yarışmasında hazırlamış olduğumuz modelin performans iyileştirmeleri için preprocessing süreçlerinin caselerini ve hyper-parametre tuning çalışmasının sonuçlarını içermektedir.

----

## BERT Tabanlı Multi Class Modellerde Hyper Parametre Tuning Araştırması

- **Model:** xlm-roberta-base
- **Max_lenght:** 64
- **Batch size:** 32
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-7 

|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.780488 | 0.925214 | 0.875000 | 0.913366 | 0.839833 |
| Recall  | 0.801670 | 0.917373 | 0.872818 | 0.887019 | 0.844538 |
| F1 Score | 0.790937 | 0.921277 | 0.873908 | 0.900000 | 0.842179 |



- **Model:** dbmdz/bert-base-turkish-uncased
- **Max_lenght:** 64
- **Batch size:** 32
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-7 

|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.880808 | 0.951681 | 0.946602 | 0.956522 | 0.962373 |
| Recall  | 0.910230 | 0.953684 | 0.965347 | 0.951923 | 0.931373 |
| F1 Score | 0.895277 | 0.952681 | 0.955882 | 0.954217 | 0.946619 |


- **Model:** dbmdz/bert-base-turkish-128k-uncased
- **Max_lenght:** 64
- **Batch size:** 32
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-6



|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.902439 | 0.976190 | 0.957921 | 0.960784 | 0.955307 |
| Recall  | 0.926931 | 0.955508 | 0.965987 | 0.942308 | 0.957983 |
| F1 Score | 0.914521 | 0.965739 | 0.961491 | 0.951456 | 0.956643 |


- **Model:** dbmdz/bert-base-turkish-128k-uncased
- **Max_lenght:** 64
- **Batch size:** 32
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-7 

|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.929787 | 0.957717 | 0.976923 | 0.953052 | 0.954167 |
| Recall  | 0.912317 | 0.957717 | 0.954887 | 0.975962 | 0.964888 |
| F1 Score | 0.920969 | 0.957717 | 0.965779 | 0.964371 | 0.959497 |



- **Model:** dbmdz/bert-base-turkish-128k-uncased
- **Max_lenght:** 64
- **Batch size:** 32
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-8


|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.914761 | 0.974194| 0.960784 | 0.958435 | 0.956885 |
| Recall  | 0.918580 | 0.959746 | 0.977556 | 0.942308 | 0.963585 |
| F1 Score | 0.916667 | 0.966916 | 0.969098 | 0.950303 | 0.960223 |



- **Model:** dbmdz/bert-base-turkish-128k-uncased
- **Max_lenght:** 64
- **Batch size:** 32
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-9



|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.898580 | 0.980392 | 0.965347 | 0.965347 | 0.952909 |
| Recall  | 0.924843 | 0.953390 | 0.972569 | 0.937500 | 0.963585 |
| F1 Score | 0.911523 | 0.966702 | 0.968944 | 0.951220 | 0.958217 |



- **Model:** dbmdz/bert-base-turkish-128k-uncased
- **Max_lenght:** 64
- **Batch size:** 16
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-7

|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.891348 | 0.971922 | 0.946210 | 0.963145 | 0.954674 |
| Recall  | 0.924843 | 0.953390 | 0.965087 | 0.942308 | 0.943978 |
| F1 Score | 0.907787 | 0.962567 | 0.955556 | 0.952612 | 0.949296 |


- **Model:** dbmdz/bert-base-turkish-128k-uncased
- **Max_lenght:** 64
- **Batch size:** 64
- **Learning rate:** 5e-5
- **Epoch:** 8
- **AdamW Eps:** 1e-7

|        | INSULT | PROFANITY | RACIST | SEXIST | OTHER |
| ------ | ------  | ------ | ------  | ------ |------ |
| Precision | 0.919421 | 0.976190 | 0.962963 | 0.958435 | 0.952909 |
| Recall  | 0.929019 | 0.955508 | 0.972569 | 0.942308 | 0.963585 |
| F1 Score | 0.924195 | 0.965739 | 0.967742 | 0.950303 | 0.958217 |

### Sonuç
Çalışmada dört farklı Türkçe BERT modeli kullanılmıştır. Bu modeller; xlm-roberta-base, dbmdz/bert-base-turkish-uncased, dbmdz/bert-base-turkish-128k-uncased (3 farklı AdamW eps seviyesi için), ve en yüksek F1 skoru veren modeldir. Farklı hiper parametreler kullanılarak sonuçları karşılaştırmak için bu modeller üzerinde çalışılmıştır. Sonuçlar, modellerin farklı performans seviyelerine sahip olduğunu ortaya koymaktadır. En iyi sonuçlar dbmdz/bert-base-turkish-128k-uncased modelinde elde edilmiştir ve bu model için F1 skoru 0.9657'dir. Bu sonuçlar, Türkçe metinler için BERT modellerinin başarılı bir şekilde kullanılabileceğini ve farklı hiper parametrelerin modellerin performansını önemli ölçüde etkileyebileceğini göstermektedir.

Sonuç olarak, çalışmada farklı Türkçe BERT modelleri ve hiper parametreleri kullanılarak, Türkçe metin sınıflandırması için en iyi performansı sağlayan model belirlenmiştir. Bu çalışma, Türkçe doğal dil işleme alanında, daha doğru ve verimli sınıflandırma modellerinin oluşturulmasına katkı sağlamaktadır.
