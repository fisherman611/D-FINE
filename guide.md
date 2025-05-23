### Step 1: Install requirements

```bash
pip install -r requirements.txt
```

### Step 2: Use this line to download data 

```bash 
wget 'https://cos.twcc.ai/fishlentrafficdataset/Fisheye8K_all_including_train%26test_update_2024Jan.zip'
```

```bash
unzip "Fisheye8K_all_including_train&test_update_2024Jan.zip" -d Fisheye8K && rm "Fisheye8K_all_including_train&test_update_2024Jan.zip"
```

### Step 3: Run Restruct data file 
```bash 
python restruct_data.py
```

### Step 4: Run split data file
```bash
python split_data.py
```

### Step 5: Set model 
```bash
export model=l  # n s m l x
```


### Step 6: Start training 
```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0
```
or 
```bash 
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

### Step 7: Testing 
```bash
CUDA_VISIBLE_DEVIVES=0 python train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --test-only -r model.pth
```


```bash 
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --test-only -r model.pth
```

