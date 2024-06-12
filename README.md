# Urluna

Urluna is a Python library for flexible array operations. It provides useful classes and functions for data preprocessing, machine learning, and various mathematical operations.

## Features

- Flexible array filling and processing
- Various operators
- Machine learning functions
- Data preprocessing tools

This library is designed to perform flexible array operations and basic matrix operations. Its functionalities include creating matrices of different types, performing operations between matrices, multiplying matrices, taking their inverses, and more.

Here are the main features of Urluna:

- Flexible Array Creation: Create 2D arrays filled with a specific value, random values, or random numbers within a specified range.
- Sequential Random Array Creation: Create a 2D array filled with random numbers but sorted in ascending order within each row.
- Range Array Creation: Create a 2D array filled with numbers within a specific range.
- Zero and One Array Creation: Create 2D arrays filled with zeros or ones of a specified size.
- Identity Matrix Creation: Create an identity matrix of a specified size.
- Various Array Operations: Perform addition, subtraction, and Hadamard product (element-wise multiplication) between arrays.
- Matrix Operations: Transpose matrices, perform multiplication and inversion between matrices, and more.
- Machine Learning Algorithms: Includes functions for Linear Regression, Random Forest, Decision Tree, and other algorithms.
- Data Preprocessing Functions: Provides tools for preprocessing data.

This library is especially designed to facilitate matrix and array operations commonly used in scientific computing, data analysis, and machine learning.
Installation

To install Urluna and learn the basics, follow these steps:

```bash
pip install urluna

```
## Usage

```bash
from urluna import *  # from urluna import Flex, Operator, MachineLearning, Preprocessing

# Flex Class
__________________________________________________________________________

The Flex class is used to perform various flexible array operations.

fill_array(value, num_rows, num_cols)

Creates a 2D array filled with the specified value.

array = Flex.fill_array(5, 3, 4)
print(array)

# Output:
# [[5 5 5 5]
#  [5 5 5 5]
#  [5 5 5 5]]
____________________________________________________________________________

sorted_random_array(min_value, max_value, num_rows, num_cols)

Creates a 2D array filled with randomly sorted values in ascending order.

array = Flex.sorted_random_array(1, 10, 3, 4)
print(array)

# Sample Output:
# [[1 4 6 9]
#  [2 5 7 8]
#  [3 5 8 9]]
_____________________________________________________________________________

random_array(min_value, max_value, num_rows, num_cols)

Creates a 2D array filled with random values.

array = Flex.random_array(1, 10, 3, 4)
print(array)

# Sample Output:
# [[3 7 4 1]
#  [8 5 2 9]
#  [6 3 7 4]]
_______________________________________________________________________________

range_array(start, stop=None, step=1)

Creates an array filled with values within a specific range.

array = Flex.range_array(1, 10, 2)
print(array)

# Output:
# [1 3 5 7 9]
_______________________________________________________________________________

zeros_array(num_rows, num_cols)

Creates a 2D array filled with zeros.

array = Flex.zeros_array(3, 4)
print(array)

# Output:
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]]
_______________________________________________________________________________

ones_array(num_rows, num_cols)

Creates a 2D array filled with ones.

array = Flex.ones_array(3, 4)
print(array)

# Output:
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
_______________________________________________________________________________

identity_matrix(size)

Creates an identity matrix of the specified size.

array = Flex.identity_matrix(4)
print(array)

# Output:
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]
#  [0 0 0 1]]
_______________________________________________________________________________

diagonal_matrix(diagonal)

Creates a diagonal matrix with the given diagonal elements.

array = Flex.diagonal_matrix([1, 2, 3, 4])
print(array)

# Output:
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]
_______________________________________________________________________________

# Operator Class

The Operator class is used to perform basic array operations.

add_arrays(array1, array2)

Adds two arrays element-wise.

array1 = Flex.ones_array(3, 3)
array2 = Flex.fill_array(2, 3, 3)
result = Operator.add_arrays(array1, array2)
print(result)

# Output:
# [[3 3 3]
#  [3 3 3]
#  [3 3 3]]
_______________________________________________________________________________

subtract_arrays(array1, array2)

Subtracts two arrays element-wise.

result = Operator.subtract_arrays(array1, array2)
print(result)

# Output:
# [[-1 -1 -1]
#  [-1 -1 -1]
#  [-1 -1 -1]]
_______________________________________________________________________________

multiply_arrays(array1, array2)

Multiplies two arrays element-wise.

result = Operator.multiply_arrays(array1, array2)
print(result)

# Output:
# [[2 2 2]
#  [2 2 2]
#  [2 2 2]]
_______________________________________________________________________________

divide_arrays(array1, array2)

Divides two arrays element-wise.

result = Operator.divide_arrays(array1, array2)
print(result)

# Output:
# [[0.5 0.5 0.5]
#  [0.5 0.5 0.5]
#  [0.5 0.5 0.5]]
_______________________________________________________________________________

```

### Note:

Usage for Preprocessing and Machine Learning will be updated soon.

### Contributing

To contribute, please create a pull request or open an issue.

### License

This project is licensed under the MIT license. For more information, see the LICENSE file.

_________________________________________________________________________________________________________________________________________________________________________________________________________


# Urluna

Urluna, esnek dizi operasyonları için bir Python kütüphanesidir. Bu kütüphane, veri ön işleme, makine öğrenimi ve çeşitli matematiksel işlemler için kullanışlı sınıflar ve fonksiyonlar sunar.

## Özellikler

- Esnek dizi doldurma ve işleme
- Çeşitli operatörler
- Makine öğrenimi fonksiyonları
- Veri ön işleme araçları

Bu kütüphane, esnek dizi işlemleri ve temel matris işlemlerini gerçekleştirmek için tasarlanmıştır. İşlevsellikler arasında, farklı türlerde matrisler oluşturmayı, matrisler arasında işlemler gerçekleştirmeyi, matrisleri çarpmayı, tersini almayı ve daha fazlasını içerir.

İşte Urluna'nın ana özellikleri:

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

Urluna'yı kurmak için aşağıdaki adımları izleyin:

```bash
pip install urluna
```

## Kullanım
```bash

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

```
### Not:

Preprocessing ve Machnine learning için kullanım yakın zamanda güncellenecektir.

### Katkıda Bulunma

Katkıda bulunmak için lütfen bir pull request oluşturun veya bir issue açın.

### Lisans
Bu proje MIT lisansı ile lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasına bakın.




