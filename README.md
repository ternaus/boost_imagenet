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
python -m src.evaluate -i PATH_TO_IMAGENET_VAL \
                       -r PATH_TO_REAL_JSON \
                       -c PATH_TO_CHECKPOINT \
                       -o vanila.json
```

and outputs file like:

```bash
{
    "accuracy": {
        "1": 81.82206375301578,
        "5": 95.0765420500886
    },
    "losses": [
        {
            "filename": "ILSVRC2012_val_00000002.JPEG",
            "loss": 0.08855907618999481
        },
        {
            "filename": "ILSVRC2012_val_00000003.JPEG",
            "loss": 0.00010000000000000009
        }
        ]
}

```

### Get images that are similar to the hard ones
Get token at [Ternaus.com/account](https://ternaus.com/account)

#### Get urls to similar images:

```bash
python -m src.get_similar_images -i PATH_TO_JSON_WITH_LOSSES \
                                 -o OUTPUT_PATH_TO_JSON_FOR_SIMILAR_IMAGES \
                                 -im PATH_TO_VAL_IMAGES \
                                 --num_images NUM_SIMILAR_IMAGES_PER_IMAGE
```

#### Download images:

```bash
python -m src.download_images \
       -a PATH_TOIMAGENET_VAL_ANNOTATIONS \
       -u OUTPUT_PATH_TO_JSON_FOR_SIMILAR_IMAGES \
       -o OUTPUT_PATH \
       -j NUM_JOBS
```

It creates a folder structure like:

```bash
n12985857
n12985857.txt
n12998815
n12998815.txt
```

where `n12985857` is a folder with images and `n12985857.txt` is a file with urls to images.
