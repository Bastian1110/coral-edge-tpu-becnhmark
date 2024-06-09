# coral-edge-tpu-becnhmark
This is just a simple repo to test how Coral AI Edge TPU performs on different computers. 

## Results 

|**Device** | **RAM** |  **Images**  | **CPU**   | **TPU** |  **Time / Image CPU**| **Time / Image TPU**|
|---|---|---|---|---|---|---|
| MacBook Air M2 | 16 GB | 5k | 53.35 s | 74.95 s | 10.67 ms | 14.90 ms |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |

## How to run this benchmark ?
1. Create a conda environment with python 3.9 
```
conda create --name coral python=3.9
```

2. Activate the environment

```
conda activate coral 
```

3. Clone this repo and cd into it

```
git clone https://github.com/Bastian1110/coral-edge-tpu-becnhmark.git && cd coral-edge-tpu-becnhmark
```

5. Install dependencies and pycoral
```
pip install -r requirements.txt
python -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

4. Download and unzip the COCO test dataset of 2017

```
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
mv test2017 images
```

5. Run the benchmark! 

```
python cpu_benchmark.py
```