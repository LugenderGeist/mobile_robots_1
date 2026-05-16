# Обход препятствий мобильным роботом
## Задание
Необходимо обеспечить управление мобильным роботом: над полем, по которому движется робот, установлена камера, транслирующая видео. Пользователем в реальном времени на видеопотоке задается конечная точка, в которую должен приехать робот; определяя препятствия и объезжая их, робот должен прибыть в конечную точку. Необходимо отобразить координаты робота во время перемещения, а также длину пройденного им пути.


## Обработка видеопотока

<div align="center">
  <img src="files/initial_video.png" alt="initial_video" height="300">
  <img src="files/aruco.png" alt="aruco" height="300">
</div>

<div align="center">
  <img src="files/obstacles.png" alt="obstacles" width="500">
</div>

В результате на видео выводятся:
- контура препятствий с запасом по расстоянию;
- контур робота;
- контур по краю кадра;
- количество препятствий;
- координаты робота;
- координаты заданной точки.

## Алгоритм и условия движения

<div align="center">
  <img src="files/dots.png" alt="dots" width="500">
</div>

<div align="center">
  <img src="files/path2.png" alt="path1" height="300">
  <img src="files/path1.png" alt="path2" height="300">
</div>

<div align="center">
  <img src="files/path3.png" alt="path1" height="300">
  <img src="files/path4.png" alt="path2" height="300">
</div>

## Заключение

