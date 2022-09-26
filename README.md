# Boost Imagenet
An example code for using [Ternaus.com](https://ternaus.com) for improving the performance of a neural network on
the ImageNet dataset.

### Save the model
```bash
import timm
import torch

model = timm.create_model('resnet34', pretrained=True)
torch.jit.save(m, MODEL_SCHECKPOINT_FILE)
```

### Evaluate on the validations set
```bash
python src/evaluate.py -i PATH_TO_IMAGENET_VAL \
                       -r PATH_TO_REAL_JSON \
                       -c PATH_TO_CHECKPOINT \
                       -o vanila.json
```
