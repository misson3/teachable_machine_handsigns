# teachable_machine_handsigns

codes for jkd-web article

Nov06, 2021

うちの実験的Web: [Teachable Machineで機械学習：Coral EdgeTPU+Raspberry piでハンドサインを認識させる](https://makeintoshape.com/teachable-machine-handsigns/)

- codes are in hand-sign-A-T-F dir



[Edge TPU Python API overview page](https://coral.ai/docs/edgetpu/api-intro/#install-the-library) says,

**This API is deprecated:** Instead try the [PyCoral API](https://coral.ai/docs/reference/py/).

But current example code from Teachable Machine imports

```python
from edgetpu.classification.engine import ClassificationEngine
```

To install the library,

```bash
sudo apt-get install python3-edgetpu

sudo dpkg -l | grep edge
ii  libedgetpu1-legacy-std:armhf    15.0    armhf    Support library for Edge TPU
ii  python3-edgetpu                      15.0    armhf    Edge TPU Python API
```

Here is how to install the latest.  Note this will kick out above libraries.

```bash
# There is newer lib for pycoral and actually the page says so.
sudo apt install python3-pycoral

sudo dpkg -l | grep edge
ii  libedgetpu1-std:armhf        16.0    armhf    Support library for Edge TPU
# libedgetpu1-legacy-std:armhf
# python3-edgetpu
# are gone!
```

