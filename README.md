## THIS REPO IS SIMPLIFIED FROM ORIGINAL REPO "MultiModal-DeepFake" FOR NOTEBOOK USE.

## Data Structure

{
        "id": 768092,
        "image": "DGM4/manipulation/HFGI/768092-HFGI.jpg",
        "text": "British citizens David and Marco BulmerRizzi in Australia celebrate the day before an event in which David won",
        "fake_cls": "face_attribute&text_attribute",
        "fake_image_box": [
            155,
            61,
            267,
            207
        ],
        "fake_text_pos": [
            8,
            13,
            17
        ],
        "mtcnn_boxes": [
            [
                155,
                61,
                267,
                207
            ],
            [
                52,
                96,
                161,
                223
            ]
        ]
    }


## Citation
```
@inproceedings{shao2023dgm4,
    title={Detecting and Grounding Multi-Modal Media Manipulation},
    author={Shao, Rui and Wu, Tianxing and Liu, Ziwei},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}

@article{shao2024dgm4++,
  title={Detecting and Grounding Multi-Modal Media Manipulation and Beyond},
  author={Shao, Rui and Wu, Tianxing and Wu, Jianlong and Nie, Liqiang and Liu, Ziwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2024},
}

```