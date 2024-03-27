def count(arr):
    print(arr)
    element_count = {}
    # Итерируемся по массиву и увеличиваем счетчик для каждого элемента
    for elem in arr:
        if elem in element_count:
            element_count[elem] += 1
        else:
            element_count[elem] = 1

    # Выводим результаты подсчета
    for key, value in element_count.items():
        print(f'Элемент {key}: {value} раз')