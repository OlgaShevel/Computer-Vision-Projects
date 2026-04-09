# Маппинг top/bottom → door2
Локальная проективная аппроксимация с взвешенной коррекцией смещения.
## Запуск
1. `pip install -r requirements.txt`
2. `python train.py` (соберёт справочники и посчитает метрику на val)
3. `python predict.py` (проверит интерфейс predict(x, y, source))
## Артефакты
`top_ref.npz`, `bottom_ref.npz` генерируются автоматически при запуске train.py.