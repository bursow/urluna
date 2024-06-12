
# Urluna

urluna, esnek dizi operasyonları için bir Python kütüphanesidir. Bu kütüphane, veri ön işleme, makine öğrenimi ve çeşitli matematiksel işlemler için kullanışlı sınıflar ve fonksiyonlar sunar.

## Özellikler

- Esnek dizi doldurma ve işleme
- Çeşitli operatörler
- Makine öğrenimi fonksiyonları
- Veri ön işleme araçları
Bu kütüphane, esnek dizi işlemleri ve temel matris işlemlerini gerçekleştirmek için tasarlanmıştır. İşlevsellikler arasında, farklı türlerde matrisler oluşturmayı, matrisler arasında işlemler gerçekleştirmeyi, matrisleri çarpmayı, tersini almayı ve daha fazlasını içerir.

İşte urluna'nın ana özellikleri:
- Esnek Dizi Oluşturma: Belirli bir değerle doldurulmuş, rastgele veya belirli bir aralıktaki rastgele sayılarla doldurulmuş 2D diziler oluşturabilirsiniz.
- Sıralı Rastgele Dizi Oluşturma: Rastgele sayılarla doldurulmuş, ancak her satırda sıralı olan bir 2D dizi oluşturabilirsiniz.
- Aralık Dizi Oluşturma: Belirli bir aralıktaki sayılarla doldurulmuş bir 2D dizi oluşturabilirsiniz.
- Sıfır ve Birlerle Dizi Oluşturma: Belirli bir boyutta sıfır veya birlerle doldurulmuş bir 2D dizi oluşturabilirsiniz.
- Kimlik Matris Oluşturma: Belirli bir boyutta birim matris oluşturabilirsiniz.
- Çeşitli Dizi İşlemleri: Diziler arasında toplama, çıkarma ve hadamard çarpımı (eleman bazında çarpma) yapabilirsiniz.
- Matris İşlemleri: Matrislerin transpozunu alabilir, matrisler arasında çarpma ve tersini alma gibi temel matris işlemleri gerçekleştirebilirsiniz.
- Machine learning'de, Linear Regression, Random Forest, Decision Tree ve diğer algoritmalar.
- Veri ön işleme fonksiyonları.
- Bu kütüphane, özellikle bilimsel hesaplamalar, veri analizi ve makine öğrenimi gibi alanlarda sıkça kullanılan matris ve dizi işlemlerini kolaylaştırmak için tasarlanmıştır.

## Kurulum

Urluna'yı kurmak için ve temelleri öğrenmek için aşağıdaki adımları izleyin:

```bash
pip install urluna ## Kurulum

____________________________Usage_____________________________

from urluna import * // from urluna import Flex, Operator, MachineLearning, Preprocessing

Flex Sınıfı
__________________________________________________________________________

Flex sınıfı, çeşitli esnek dizi işlemleri gerçekleştirmek için kullanılır.
fill_array(value, num_rows, num_cols)

Belirtilen değerle doldurulmuş bir 2D dizi oluşturur.

array = Flex.fill_array(5, 3, 4)
print(array)

# Çıktı:
# [[5 5 5 5]
#  [5 5 5 5]
#  [5 5 5 5]]
____________________________________________________________________________

sorted_random_array(min_value, max_value, num_rows, num_cols)

Artan sırada sıralı rastgele değerlerle doldurulmuş bir 2D dizi oluşturur.

array = Flex.sorted_random_array(1, 10, 3, 4)
print(array)

# Örnek Çıktı:
# [[1 4 6 9]
#  [2 5 7 8]
#  [3 5 8 9]]
_____________________________________________________________________________

random_array(min_value, max_value, num_rows, num_cols)

Rastgele değerlerle doldurulmuş bir 2D dizi oluşturur.

array = Flex.random_array(1, 10, 3, 4)
print(array)

# Örnek Çıktı:
# [[3 7 4 1]
#  [8 5 2 9]
#  [6 3 7 4]]
_______________________________________________________________________________

range_array(start, stop=None, step=1)

Belirli bir aralıktaki değerlerle doldurulmuş bir dizi oluşturur.

array = Flex.range_array(1, 10, 2)
print(array)

# Çıktı:
# [1 3 5 7 9]
_______________________________________________________________________________

zeros_array(num_rows, num_cols)

Sıfırlarla doldurulmuş bir 2D dizi oluşturur.

array = Flex.zeros_array(3, 4)
print(array)

# Çıktı:
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]]
_______________________________________________________________________________

ones_array(num_rows, num_cols)

Birlerle doldurulmuş bir 2D dizi oluşturur.

array = Flex.ones_array(3, 4)
print(array)

# Çıktı:
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
_______________________________________________________________________________

identity_matrix(size)

Belirtilen boyutta birim matris oluşturur.

array = Flex.identity_matrix(4)
print(array)

# Çıktı:
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]
#  [0 0 0 1]]
_______________________________________________________________________________

diagonal_matrix(diagonal)

Verilen diyagonal elemanlarla bir diyagonal matris oluşturur.

array = Flex.diagonal_matrix([1, 2, 3, 4])
print(array)

# Çıktı:
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]
_______________________________________________________________________________

Operator Sınıfı

Operator sınıfı, temel dizi işlemleri gerçekleştirmek için kullanılır.
add_arrays(array1, array2)

İki diziyi eleman bazında toplar.

array1 = Flex.ones_array(3, 3)
array2 = Flex.fill_array(2, 3, 3)
result = Operator.add_arrays(array1, array2)
print(result)

# Çıktı:
# [[3 3 3]
#  [3 3 3]
#  [3 3 3]]
_______________________________________________________________________________

subtract_arrays(array1, array2)

İki diziyi eleman bazında çıkarır.


result = Operator.subtract_arrays(array1, array2)
print(result)

# Çıktı:
# [[-1 -1 -1]
#  [-1 -1 -1]
#  [-1 -1 -1]]
_______________________________________________________________________________

multiply_arrays(array1, array2)

İki diziyi eleman bazında çarpar.


result = Operator.multiply_arrays(array1, array2)
print(result)

# Çıktı:
# [[2 2 2]
#  [2 2 2]
#  [2 2 2]]
_______________________________________________________________________________

divide_arrays(array1, array2)

İki diziyi eleman bazında böler.


result = Operator.divide_arrays(array1, array2)
print(result)

# Çıktı:
# [[0.5 0.5 0.5]
#  [0.5 0.5 0.5]
#  [0.5 0.5 0.5]]
_______________________________________________________________________________

NOT:

Preprocessing ve Machnine learning için kullanım yakın zamanda güncellenecektir.

Katkıda Bulunma

Katkıda bulunmak için lütfen bir pull request oluşturun veya bir issue açın.
Lisans

Bu proje MIT lisansı ile lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına bakın.




