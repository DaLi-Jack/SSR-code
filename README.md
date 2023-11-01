### Install
```bash
conda create -n ssr python=3.8
conda activate ssr
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Training
```bash
# NOTE: set show_rendering=False

# for 3D-FRONT dataset
python train.py --config configs/train_front3d.yaml

# for Pix3D dataset
python train.py --config configs/train_pix3d.yaml
```

### Testing
**1.export mesh with color and in original size of object**
```bash
# NOTE: set show_rendering=False, eval.export_mesh=True, eval.export_color_mesh=True

# for 3D-FRONT dataset
python inference.py --config configs/train_front3d.yaml

# for Pix3D dataset
python inference.py --config configs/train_pix3d.yaml

# for SUNRGB-D dataset
python inference_sunrgbd.py --config configs/train_sunrgbd.yaml
```

**2.novel view synthesis**
```bash
# NOTE: set show_rendering=True
python inference_rot_angle.py --config configs/train_front3d.yaml
```

**3.scene fusion**
```bash
# NOTE: set show_rendering=True, eval.fusion_scene=True
# and remember to change data.batch_size.test to object number !!!
# please carefully compare the differences between train_front3d.yaml and train_front3d_fusion.yaml
python inference_rot_angle.py --config configs/train_front3d_fusion.yaml
```

### Evaluation
```bash
# NOTE: set eval.export_mesh=True, eval.export_color_mesh=False
bash eval/evaluate.sh configs/train_front3d.yaml
```

