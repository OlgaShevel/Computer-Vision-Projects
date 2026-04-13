# Маппинг top/bottom → door2
Локальная проективная аппроксимация с взвешенной коррекцией смещения.
### Запуск
1. `pip install -r requirements.txt`
2. `python train.py` (соберёт справочники и посчитает метрику на val)
3. `python predict.py` (проверит интерфейс predict(x, y, source))
### Артефакты
`top_ref.npz`, `bottom_ref.npz` генерируются автоматически при запуске train.py.



============================

Визуальные примеры - в коде этого нет.

### Пример расчета евклидова расстояния (L2) для всех точек пары top-door2 сюжета на val

![4](https://github.com/user-attachments/assets/037f788f-d750-48d5-8218-143fc818f224)


### Визуализация 

1) Для одной точки

![2](https://github.com/user-attachments/assets/2a598dde-f403-4f8f-8b80-7168e1690ad0)


2) Для всех точек пары
   
   (На фото door2 пурпурные точки - L2 < 100, точки циан - L2 > 100, белые точки - предсказанные)  

![3](https://github.com/user-attachments/assets/71480c8e-c0f9-454a-8dae-91ac97579a06)
