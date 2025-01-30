## Notes for getting set up on Lambda Labs machine:

Sync local to remote - set up ssh.
e.g. using PyCharm - set up Pycharm remote deployment.
Ensure all source files are copied to remote.

install datasets
```
./datasets/install.sh
```

install pytorch etc. (required special install)
```
pip install torch torchvision torchaudio
```

install dependencies
```
pip install -r cogeneration/requirements.txt
```

install cogeneration package as editable
```
pip install -e cogeneration/
```

setup wandb
```
wandb login
```

train with logging
```
cd cogeneration
python cogeneration/scripts/train.py 2>&1 | tee train.log
```

copy log to local if needed
```
scp ubuntu@lambda_labs_tester:/home/ubuntu/cogeneration/train.log train.log
```