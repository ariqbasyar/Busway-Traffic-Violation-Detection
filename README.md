# Busway-Traffic-Violation-Detection

This project is intended for the fulfillment of final project of
Advanced Computer Networks 2021/2022 course Universitas Indonesia.

Group Members:
* [Muhammad Ariq Basyar](https://github.com/ariqbasyar/)
* [Douglas Raevan Faisal](https://github.com/douglasraevan/)

Acks:
* https://github.com/NEFTeam/Traffic-Law-Enforcement
* https://github.com/ultralytics/yolov5

Installations:
* Use the latest pip version

    ```bash
    pip install --upgrade pip
    ```

* Install the requirements

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

* Run example script to test your configs

    ```bash
    python example.py
    ```

* Fog and Server

    ```bash
    python service/server.py
    python service/fog.py
    ```

    > Make sure the variable `_type` (in [fog.py](service/fog.py) and
    [server.py](service/server.py)) has the same value for the both fog and
    server.

References:
* Omidi, A., Heydarian, A., Mohammadshahi, A., Beirami, B. A., & Haddadi, F. (2021). An Embedded Deep Learning-based Package for Traffic Law Enforcement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 262-271).
