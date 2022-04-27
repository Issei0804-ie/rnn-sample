build-sif:
	singularity build --fakeroot output/tensorflow-gpu.sif tensorflow-gpu.def 

build-model:
	singularity run --nv output/tensorflow-gpu.sif python temp.py

run:
	singularity run --nv output/tensorflow-gpu.sif python generate.py

pip-install:
	singularity run output/tensorflow-gpu.sif pip install -r requirements.txt

sync-amane:
	rsync -avhz main.py amane:~/cnn/main.py
	rsync -avhz cnn.py amane:~/cnn/cnn.py
	rsync -avhz nn.py amane:~/cnn/nn.py

