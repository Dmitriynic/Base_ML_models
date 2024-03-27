import pandas as pd
import random
import time
start_time = time.time()
# Фиксированный seed для воспроизводимости результатов
random.seed(42)

def generate_data(num_rows):
    data = list(range(num_rows))
    return data

def process_data(data):
    processed_data = [(value, value ** 2) for value in data]
    return pd.DataFrame(processed_data, columns=["value", "square"])

if __name__ == "__main__":
    # Генерация данных
    num_rows = 50000000
    data = generate_data(num_rows)

    # Обработка данных
    processed_df = process_data(data)
    print('time', time.time() - start_time)

    # Вывод результата
    print(processed_df)