# Sıfırdan Sinir Ağı (C++)

Hiçbir harici makine öğrenmesi kütüphanesi (NumPy, PyTorch, Eigen vb.) kullanmadan, saf C++ ve `std::vector` ile yazılmış bir ileri beslemeli sinir ağı (feedforward neural network). Forward propagation, backpropagation ve gradient descent tamamen elle implemente edilmiştir. README.md dosyasında yapay zekadan faydalanılsa da projenin geneli elle yazılmıştır.

Bu proje öğrenme amaçlıdır: bir sinir ağının nasıl çalıştığını adım adım göstermek için yazılmıştır.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Mimari](#mimari)
- [Temel Kavramlar](#temel-kavramlar)
- [Kod Yapısı](#kod-yapısı)
- [Kullanım](#kullanım)
- [Eğitim Akışı](#eğitim-akışı)
- [Aktivasyon Fonksiyonları](#aktivasyon-fonksiyonları)
- [Bilinen Tuzaklar ve Çözümler](#bilinen-tuzaklar-ve-çözümler)
- [Geliştirme Önerileri](#geliştirme-önerileri)

## Genel Bakış

Ağ, katmanların bir dizisi olarak kurulur (örneğin `4 → 3 → 2 → 1`). Her katman bir önceki katmandan tam bağlantılı (fully connected) ağırlıklarla beslenir. Eğitim, mini-batch destekli stokastik gradient descent ile yapılır.

Desteklenen özellikler:

- Tam bağlantılı (dense) katmanlar
- Sigmoid, ReLU ve lineer (identity) aktivasyonlar
- MSE (Mean Squared Error) kayıp fonksiyonu
- Mini-batch eğitim
- Manuel backpropagation (zincir kuralı ile)
- Gradient checking (sayısal gradyan ile doğrulama)

## Mimari

```
Input Layer        Hidden Layers              Output Layer
[x1]                 [h1]      [o1]
[x2]   ──weights──>  [h2]  ──> [o2]  ──weights──>  [y]
[x3]                 [h3]
[x4]
```

Veri matrisi `[örnek_sayısı] × [feature_sayısı]` boyutundadır:

- **Satır** = bir örnek (sample)
- **Sütun** = bir feature
- **Input neuron sayısı** = feature sayısı (sütun sayısı)

Örnek: `15 × 3`'lük bir veri, 3 input neuronu ve 15 eğitim örneği demektir.

## Temel Kavramlar

### Forward Propagation

Her neuron için, bir önceki katmanın çıktılarının ağırlıklı toplamı hesaplanır, üzerine aktivasyon uygulanır:

```
z = Σ (w_i · x_i) + bias
a = f(z)          // f: aktivasyon fonksiyonu
```

### Backpropagation

Kayıp fonksiyonundan başlayarak zincir kuralı ile gradyanlar geriye doğru yayılır.

**Tohum (seed):** En sondaki çıktı neuronunun gradyanı, kayıp fonksiyonunun türevidir. MSE için:

```
output.grad = 2 · (tahmin − gerçek)
```

**Her katman için (sondan başa):**

```
delta_j   = neuron_j.grad · f'(z_j)              // aktivasyon türevi
w.grad   += delta_j · (önceki neuronun çıktısı)  // ağırlık gradyanı
x.grad   += Σ (delta_j · w)                       // önceki katmana yayılacak gradyan
```

> **Önemli:** Aktivasyon türevi `f'`, ağırlığın **gittiği** (çıkış tarafındaki) neurona aittir, geldiği neurona değil.

### Gradient Descent

Tüm gradyanlar hesaplandıktan **sonra** ağırlıklar güncellenir:

```
w = w − learning_rate · w.grad
```

Güncelleme her örnekte değil, batch bittikten sonra bir kez yapılır.

## Kod Yapısı

### `struct Value`

Bir ağırlığı temsil eder: değeri (`data`) ve gradyanını (`grad`) tutar.

### `struct Neuron`

Bir neuronu temsil eder: ham toplam (`value`), aktivasyon sonrası değer (`activation_value`) ve gradyan (`grad`).

### `class Input`

Eğitim verisini (`vector<vector<double>>`) saran basit bir sarmalayıcı.

### `class Layer`

Tek bir katmanı temsil eder:

- `weights` — ağırlık matrisi (`[current] × [previous]`)
- `neurons` — katmandaki neuronlar
- `previous`, `current` — giriş ve çıkış neuron sayıları
- `first_weights()` — ağırlıkları başlatır

### `class Sequence`

Eğitim motorunun tamamını içerir:

| Fonksiyon | Görevi |
|-----------|--------|
| `activation_func` | İleri yönde aktivasyon (sigmoid / ReLU / lineer) |
| `activation_der` | Aktivasyon türevi (backprop için) |
| `forward` | Bir katmanın ileri hesabı |
| `backward` | Bir katmanın gradyan hesabı |
| `optimizer` | Ağırlık güncellemesi |
| `loss_der` | Kayıp türevi (tohum) |
| `zero_grad` | Tüm gradyanları sıfırlar |
| `one_step` | Tek bir örnek için forward + backward |
| `all_stages` | Tüm epoch ve batch döngüsü |

## Kullanım

```cpp
int main(){
    // Eğitim verisi: 3 örnek, her biri 3 feature
    vector<vector<double>> v = {{1,2,3}, {2,3,5}, {3,4,7}};
    vector<double> y = {6, 10, 14};   // hedef değerler

    Input input(v);
    int first_size = v[0].size();

    // Katmanlar: 3 → 3 → 2 → 1
    Layer layer1 = Layer(0, first_size);   // input katmanı
    Layer layer  = Layer(first_size, 3);
    Layer layer2 = Layer(3, 2);
    Layer layer3 = Layer(2, 1);

    vector<Layer> layers = {layer1, layer, layer2, layer3};

    // Sequence(katmanlar, veri, batch_size, hedefler, kayıp, epoch, lr, aktivasyon)
    Sequence model(layers, input, 1, y, "MSE", 5000, 0.001, "ReLU");

    return 0;
}
```

### Parametreler

- **batch_size** — bir güncellemede kaç örnek işleneceği
- **loss** — `"MSE"`
- **epoch** — tüm verinin kaç kez geçileceği
- **lr** — learning rate (öğrenme oranı)
- **activate** — gizli katman aktivasyonu: `"sigmoid"`, `"ReLU"` veya lineer için diğer

## Eğitim Akışı

```
her epoch için:
    her batch için:
        zero_grad()                    // gradyanları sıfırla
        her örnek için (batch içinde):
            forward    (tüm katmanlar)
            seed       (output.grad = loss türevi)
            backward   (tüm katmanlar, gradyanlar += ile birikir)
        optimizer (bir kez, biriken gradyanlarla güncelle)
```

## Aktivasyon Fonksiyonları

| Aktivasyon | Fonksiyon | Türev | Kullanım |
|-----------|-----------|-------|----------|
| Sigmoid | `1 / (1 + e^-z)` | `a · (1 − a)` | İkili sınıflandırma çıkışı |
| ReLU | `max(0, z)` | `z > 0 ? 1 : 0` | Gizli katmanlar (önerilen) |
| Lineer | `z` | `1` | Regresyon çıkışı |

> **Kural:** Gizli katmanlarda ReLU, regresyon çıkış katmanında lineer kullan. Çıkış katmanında sigmoid kullanırsan çıktı 0–1 ile sınırlanır ve 6, 10, 14 gibi değerleri üretemez.

## Bilinen Tuzaklar ve Çözümler

Bu proje geliştirilirken karşılaşılan ve çözülen yaygın hatalar:

### 1. Ağırlıkların hepsini aynı değerle başlatmak (simetri problemi)

Tüm ağırlıkları `1.0` ile başlatırsan, bir katmandaki neuronlar birbirinin tıpatıp aynısı olur ve öyle kalır — ağ kapasitesini kaybeder. **Çözüm:** küçük rastgele değerlerle başlat:

```cpp
static mt19937 gen(42);
normal_distribution<double> dist(0.0, 0.1);
v.data = dist(gen);
```

### 2. Çıkış katmanında sigmoid (regresyon için)

Sigmoid çıktısı her zaman 0–1 arasındadır. Hedefler 6, 10, 14 ise ağ asla ulaşamaz, 0.1 civarında takılır. **Çözüm:** çıkış katmanını lineer yap.

### 3. Vanishing gradient (sigmoid)

Sigmoid türevinin maksimumu 0.25'tir; derin ağlarda gradyanlar üstel olarak küçülür. **Çözüm:** gizli katmanlarda ReLU kullan.

### 4. Exploding gradient / NaN

Büyük learning rate veya büyük başlangıç ağırlıkları, ağırlıkları `inf`'e, oradan `NaN`'a taşır. **Çözüm:** learning rate'i düşür (`0.0001`), ağırlıkları küçük başlat, gerekirse gradient clipping uygula.

### 5. Forward/backward sıralaması

Forward'ın **tamamı** bitmeden backward başlatılamaz. Üç aşama ayrı döngülerde olmalı: önce tüm forward, sonra seed, sonra tüm backward (sondan başa), en son optimizer.

### 6. Gradyan tohumu (seed) unutmak

Backward'dan önce çıkış neuronunun gradyanını kayıp türeviyle set etmezsen, tüm gradyanlar 0 (ya da çöp) olur:

```cpp
l[size_l-1].neurons[0].grad = loss_der(pred, y, "MSE");
```

### 7. `zero_grad` unutmak

Ağırlık gradyanları `+=` ile birikir. Her batch başında hem neuron hem ağırlık gradyanlarını sıfırlamazsan, gradyanlar üst üste birikip patlar.

### 8. Forward/backward aktivasyon tutarsızlığı

Çıkış katmanını forward'da lineer (`"simple"`) çağırıp backward'da ReLU türeviyle hesaplarsan tutarsızlık olur. İkisinde de aynı aktivasyonu kullan.

## Doğrulama: Gradient Checking

Backpropagation'ın doğru olduğunu, gradyanı iki bağımsız yolla hesaplayıp karşılaştırarak doğrulayabilirsin:

```cpp
// Analitik (backprop'un bulduğu)
double analytical = l[1].weights[0][0].grad;

// Sayısal (sadece forward + loss ile)
double eps = 1e-4;
double saved = l[1].weights[0][0].data;

double loss0 = pow(pred - y, 2);                  // loss(w)
l[1].weights[0][0].data = saved + eps;
// forward'ı baştan çağır...
double loss1 = pow(pred_new - y, 2);              // loss(w+eps)
l[1].weights[0][0].data = saved;

double numerical = (loss1 - loss0) / eps;
// numerical ≈ analytical olmalı (aynı işaret, yakın değer)
```

İkisi yakınsa backprop doğru; ters işaretliyse bir yerde işaret hatası, çok farklıysa formül hatası vardır.

## Geliştirme Önerileri

- **Bias desteği** — şu an bias kısmen var; tam entegre edilebilir
- **Veri normalizasyonu** — girdileri ölçeklemek eğitimi stabilize eder
- **Farklı kayıp fonksiyonları** — Cross-entropy (sınıflandırma için)
- **Gelişmiş optimizer'lar** — Momentum, Adam
- **Veri karıştırma (shuffle)** — her epoch başında örnek sırasını karıştırmak
- **Performans** — `-O3 -march=native` derleme bayrakları, sonra OpenMP ile paralelleştirme
- **Katmanları referansla geçirmek** — `Sequence` constructor'ı katmanları kopyalamak yerine referansla almalı

## Derleme

```bash
g++ -O2 -std=c++17 main.cpp -o nn
./nn
```

Hız optimizasyonu için:

```bash
g++ -O3 -march=native -std=c++17 main.cpp -o nn
```

---

*Bu proje, sinir ağlarının iç çalışma mantığını anlamak için sıfırdan yazılmıştır. Üretim ortamı için PyTorch, TensorFlow gibi optimize edilmiş kütüphaneler önerilir.*
