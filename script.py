import pandas as pd
from flask import Flask, request, jsonify
import json

# Загрузка модели CatBoost
from catboost import CatBoostRegressor
model = CatBoostRegressor()

# Необходимо указать путь!
model.load_model('realty_model')

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Получение данных из запроса
    data = request.get_json()

    # Преобразование данных в DataFrame
    df = pd.read_json(data, orient ='index').T
    
    # Подготовка данных
    data = prepare_data(df)

    # Выполнение предсказания с помощью модели CatBoost
    predictions = model.predict(data)

    # Формирование ответа в формате JSON
    response = {
        'predictions': predictions.tolist()
    }

    return jsonify(response)

# Подготовка данных
def prepare_data(data):
    
    # Переименование
    data.columns = ['date_first', 'date_last', 'text_box', 'type',
                    'location', 'street', 'building_number', 'floor',
                    'plan', 'all_area', 'living_area', 'kitchen_area',
                    'x1', 'x2', 'x3', 'x4', 'x5']
    
    #print(data.shape)
    
    data['floor'] = data['floor'].str.extract(r'(\d+/\d+)')
    data = data.dropna(subset=['floor'])
    
    #print(data.shape)
    
    # В нижний регистр
    data['location'] = data['location'].str.lower()
    data['type'] = data['type'].str.lower()
    
    # Переименование редких районов
    main_locations = ['орджоникидзевский', 'ленинский',
                      'орджоникидзевский (левый берег)',
                      'ленинский (левый берег)', 'правобережный']
    data['location'] = data['location'].apply(lambda x: x if x in main_locations else 'другой')

    # Тип квартиры в количество комнат
    type_mapping = {
    'однокомнатная': 1,
    'двухкомнатная': 2,
    'трехкомнатная': 3,
    'четырехкомнатная': 4,
    'многокомнатная' : 5
    }
    data['rooms'] = data['type'].map(type_mapping)
    
    # Новые столбцы
    data['date_last'] = pd.to_datetime(data['date_last'])
    data['month'] = data['date_last'].dt.month
    data['year'] = data['date_last'].dt.year
    data[['floor_act', 'floor_max']] = data['floor'].str.split('/', expand=True)
    
    # Удаляю лишнее
    data.drop(['date_first', 'date_last', 'text_box', 'type',
               'street', 'building_number', 'floor', 
               'plan', 'x1', 'x2', 'x3', 'x4', 'x5'], axis=1, inplace=True)
    data.dropna(inplace=True)
    
    #print(data.shape)
    
    # Меняю типы данных
    data['floor_act'] = data['floor_act'].astype('int')
    data['floor_max'] = data['floor_max'].astype('int')
    data['all_area'] = data['all_area'].astype('float')
    data['living_area'] = data['living_area'].astype('float')
    data['kitchen_area'] = data['kitchen_area'].astype('float')
    data['location'] = data['location'].astype('category')
    data['month'] = data['month'].astype('int')
    data['year'] = data['year'].astype('int')
    data['rooms'] = data['rooms'].astype('int')
    #print(data.shape)
    
    # Удаляю дубликаты
    data.drop_duplicates(inplace=True, ignore_index=True)
    
    # кодирование признаков
    data_encoded = pd.get_dummies(data, columns=['location', 'year', 'month'])
    
    columns = ['all_area', 'living_area', 'kitchen_area', 'rooms', 'floor_act',
       'floor_max', 'location_другой', 'location_ленинский',
       'location_ленинский (левый берег)', 'location_орджоникидзевский',
       'location_орджоникидзевский (левый берег)', 'location_правобережный',
       'year_2022', 'year_2023', 'month_1', 'month_2', 'month_3', 'month_4',
       'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
       'month_11', 'month_12']
    
    empty_df = pd.DataFrame(columns=columns)
    result_df = pd.concat([empty_df, data_encoded], ignore_index=True)
    result_df.fillna(0, inplace=True)
    
    return result_df

if __name__ == '__main__':
    app.run()
